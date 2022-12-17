import konlpy 

from typing import List
from konlpy.tag import Mecab
from functools import reduce

mecab = Mecab()

def concat_pos(pos_list : List[tuple], concat_pos_list : List[tuple]) -> List[tuple]:
    if not pos_list:
        return concat_pos_list

    if len(concat_pos_list)>0 and concat_pos_list[-1][1].startswith(('XSV', 'VV', 'VA')) and pos_list[0][1] in ['EC', 'EP', 'EF', 'ETM']:
        concat_pos_list[-1] = (concat_pos_list[-1][0]+pos_list[0][0],
                            f"{concat_pos_list[-1][1]}+{pos_list[0][1]}")
        return concat_pos(pos_list[1:], concat_pos_list)
    
    if len(concat_pos_list)>0 and concat_pos_list[-1][1].startswith(('NNG', 'NNP', 'XR', 'SN')) and pos_list[0][1] in ['NNG', 'NNP', 'XR', 'NNBC']:
        concat_pos_list[-1] = (concat_pos_list[-1][0]+pos_list[0][0],
                            f"{concat_pos_list[-1][1]}+{pos_list[0][1]}")
        return concat_pos(pos_list[1:], concat_pos_list)
    
    return concat_pos(pos_list[1:], concat_pos_list + [pos_list[0]])

def analyze_syntactics(sent : str, split_morphs : bool) -> tuple:
    pos_list = mecab.pos(sent)

    pos_list = concat_pos(pos_list, [])
    noun_list = [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR', 'SN', 'NNBC'))]
    
    verb_list = [pos for pos, tag in pos_list if tag.startswith(('XSV', 'VV', 'VA'))]

    if not split_morphs:
        return [pos for pos, tag in pos_list if tag.startswith(('NNG', 'NNP', 'XR', 'SN', 'NNBC', 'XSV', 'VV', 'VA'))]
    return noun_list, verb_list

def extract_keyword(persona : List[str]) -> str:
    kw_list = list(map(lambda x: analyze_syntactics(x, False), persona))
    return list(reduce(lambda x, y: x + y, kw_list))