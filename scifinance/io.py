import os
import json
import numpy as np

from typing import List


def read_json(path: str) -> List[dict]:
    objects = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            objects.append(obj)
        f.close()
    return objects


def write_json(objects: List[dict] or np.array(dict), path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for obj in objects:
            line = json.dumps(obj, ensure_ascii=False)
            f.write(line + '\n')
        f.close()
    return


# if __name__ == '__main__':
#     r_path = '标注样例.json'
#     w_path = '标注样例2.json'
#     objects = read_json(r_path)
#     print(objects[0])
#     print(type(objects[0]))
#     write_json(objects, w_path)
