# -- coding: utf-8 --

from flask import Flask, render_template, Response, make_response
from flask_restful import Resource, Api
from flask_restful import reqparse

from functools import wraps

import dmn_model as DMN

import json

app = Flask(__name__)
api = Api(app)


context_1='''1 지급액은 1000만원 - 이미 지급된 건강진단보험금입니다.
2 유족 위로금에 관한 항목입니다. 
3 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다. 
4 지급액은 300만원입니다.
5 보험기간(종신) 중 피보험자가 사망하였을 때:
6 사망보험금 및 유족위로금
7 사망 보험금에 관한 항목입니다.
8 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다.
'''
context_2='''1 지급액은 1500만원 - 이미 지급된 건강진단보험금입니다.
2 유족 위로금에 관한 항목입니다. 
3 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다. 
4 지급액은 350만원입니다.
5 보험기간(종신) 중 피보험자가 사망하였을 때:
6 사망보험금 및 유족위로금
7 사망 보험금에 관한 항목입니다.
8 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다.
'''
context_3='''1 지급액은 3000만원 - 이미 지급된 건강진단보험금입니다.
2 유족 위로금에 관한 항목입니다. 
3 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다. 
4 지급액은 900만원입니다.
5 보험기간(종신) 중 피보험자가 사망하였을 때:
6 사망보험금 및 유족위로금
7 사망 보험금에 관한 항목입니다.
8 보험기간 중 피보험자가 사망하였을 때 지급사유가 인정됩니다.
'''


def json_api(f):
    @wraps
    def inner(*args, **kwds):
        r = f(*args, **kwds)
        result = make_response(json.dumps(r, ensure_ascii=False))
        return result
    return inner


@app.route("/")
def template_main():    
    return render_template('index.html')


class read_qna(Resource):
    def post(self):
        try:   
            parser = reqparse.RequestParser()
            parser.add_argument('question', type=str)
            parser.add_argument('context', type=str)
            args = parser.parse_args()

            # model을 통해 답변 요청
            _question = args['question']
            _context = args['context']
            
            contexts_idx = _context.split('-')[-1]
            context = None
            ## context 1, 2, 3 ##
            if contexts_idx == '1':
                context = context_1
            elif contexts_idx == '2':
                context = context_2
            elif contexts_idx == '3':
                context = context_3

            MODEL = DMN.dmn_model()
            MODEL.load_glove()
            MODEL.load_graph()
            
            _input_str = context_1 + '7 ' + _question 
            if contexts_idx == 1:
                _answer = MODEL.getAnswer(_input_str, _input_str)
            else:
                _input_str_new = context + '7 ' + _question
                _answer = MODEL.getAnswer(_input_str, _input_str_new)

            _response_dic = {'question':_question, 'context': _context, 'answer': _answer[0]}
            _response_json = json.dumps(_response_dic, ensure_ascii=False)
            _res = make_response(_response_json)
            _res.headers['Content-Type'] = 'application/json'

            return _res

        except Exception as e:
            return {'error', str(e)}

api.add_resource(read_qna, '/read_qna')

if __name__ == '__main__':
    app.run(debug=True)
