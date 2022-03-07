from flask import Flask, request, Response, abort
import traceback
import logging
from model import compare_doc

app = Flask(__name__)


@app.route('/compare_data', methods=['POST'])
def compare_data():
    try:
        docs = request.json

        first = docs.get('First')
        second = docs.get('Second')

        if(first is None or second is None):
            raise Exception("Docs names not correct")

        # send data to model
        result = compare_doc(first,second)

        return result
    except Exception as e:
        logging.error(traceback.format_exc())
        return Response(response="Bad Request", status=400)
