import torch
import torchcrf
from transformers import BertModel


class NER(torch.nn.Module):
    r"""
        Args:
            bert_path: The folder contains your bert settings.
            num_tags: Nums of tags.
        Inputs:
            **inputs** of shape `(batch_size, seq_len)`: Preprocessed sequences.
            **targets** of shape `(batch_size, seq_len)`: Preprocessed tags.
        Outputs:
            The negative log likelihood of the best tags.
            We can use it as the loss.
    """

    def __init__(self, bert_path, out_num_tags):
        super().__init__()
        self.bert_path = bert_path
        self.num_tags = out_num_tags
        self.embedding_dim = 768
        self.hidden_dim = 128

        self.bert = BertModel.from_pretrained(self.bert_path)
        self.lstm = torch.nn.GRU(
            input_size=self.embedding_dim,
            output_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        self.dense = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_tags,
            bias=False
        )
        self.crf = torchcrf.CRF(
            num_tags=self.num_tags,
            batch_first=True
        )

        # Frozen the bert model
        for param in self.bert.parameters():
            param.requires_grad = False

    def _emission_matrix(self, inputs):
        bert_embed = self.bert(inputs)[0]
        lstm_outputs, _ = self.lstm(bert_embed)
        emissions = self.dense(lstm_outputs)
        return emissions

    def forward(self, inputs, targets):
        emissions = self._emission_matrix(inputs)
        log_likelihood = self.crf(emissions, targets, reduction='mean')
        return -log_likelihood

    def predict(self, inputs):
        r"""
            Args:
                **inputs** of shape `(batch_size, seq_len)`: Preprocessed sequences.
            Outputs:
                **List** of shape `(batch_size, seq_len)`:
                    Each contains the numbers of tags predicted
        """
        emits = self._emission_matrix(inputs)
        tags_eval = self.crf.decode(emits)
        return tags_eval


class Fusion(torch.nn.Module):
    r"""
        Args:
            input_dim1: The third dimension of input1
            input_dim2: The third dimension of input2
            output_dim: The third dimension of outputs
        Inputs:
            **input1** of shape `(batch_size, seq_len, dim1)`
            **input2** of shape `(batch_size, seq_len, dim2)`
        Outputs:
            **outputs** of shape `(batch_size, seq_len, dim)`
    """

    def __init__(self, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.input_len = input_dim1 + input_dim2
        self.output_len = output_dim
        self.dense = torch.nn.Linear(
            in_features=self.input_len,
            out_features=self.output_len,
            bias=False
        )

    def forward(self, input1, input2):
        inputs = torch.cat((input1, input2), 2)
        outputs = self.dense(inputs)
        return outputs


class SelfAttention(torch.nn.Module):
    r"""
        Args:
            embed_dim: The embedding dim of the sequence.
        Inputs:
            **inputs** of shape `(batch_size, seq_len, embed_dim)`
        Outputs:
            **attn_output** of shape `(batch_size, seq_len, embed_dim)`
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=1
        )

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        attn_output, _ = self.attention(inputs, inputs, inputs)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class AdvancedNER(torch.nn.Module):
    r"""
        Args:
            bert_path: The path of bert model
            num_tags: The number of output tags
            pos_num_tags: The number of parts of speech
        Inputs:
            **input1** of shape `(batch_size, seq_len)`
            **input2** of shape `(batch_size, seq_len, pos_num_tags)`
            **target_tags** of shape `(batch_size, seq_len)`
        Outputs:
            The negative log likelihood of the best tags.
            We can use it as the loss.
    """

    def __init__(self, bert_path, out_num_tags, pos_num_tags):
        super().__init__()
        self.bert_path = bert_path
        self.num_tags = out_num_tags
        self.pos_num_tags = pos_num_tags
        self.embedding_dim = 768
        self.hidden_dim = 128

        self.bert = BertModel.from_pretrained(self.bert_path)
        self.lstm = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        self.dense = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_tags,
            bias=False
        )
        self.attn = SelfAttention(pos_num_tags)
        self.fusion = Fusion(
            input_dim1=self.num_tags,
            input_dim2=self.pos_num_tags,
            output_dim=self.num_tags
        )
        self.crf = torchcrf.CRF(
            num_tags=self.num_tags,
            batch_first=True,
        )

        for param in self.bert.parameters():
            param.requires_grad = False

    def _emission_matrix(self, input1, input2):
        bert_embed = self.bert(input1)[0]
        lstm_outputs, _ = self.lstm(bert_embed)
        dense_output = self.dense(lstm_outputs)
        attn_output = self.attn(input2)
        emissions = self.fusion(dense_output, attn_output)
        return emissions

    def forward(self, input1, input2, targets):
        emissions = self._emission_matrix(input1, input2)
        log_likelihood = self.crf(emissions, targets, reduction='mean')
        return -log_likelihood

    def predict(self, input1, input2):
        emits = self._emission_matrix(input1, input2)
        tags_eval = self.crf.decode(emits)
        return tags_eval
