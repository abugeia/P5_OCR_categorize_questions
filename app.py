# -*- coding: utf-8 -*-

from flask import Flask, jsonify, render_template, request
import lib

app = Flask(__name__)

############################################
# Endpoints
############################################


@app.route("/", methods=['GET', 'POST'])
def HelloWorld():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"

    # Return the response in json format
    return jsonify(response)


@app.route("/predict/", methods=['GET', 'POST'])
def tags():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("tag-gen.html", href='Type a question')
    else:
        text = request.form['question'] + ' ' + request.form['titre']
        model, sw, lb, vec = lib.import_model()
        tags = lib.treat_text_get_tags(text, model, lb, vec, sw)
        return render_template("tag-gen.html", href=' '.join(tags[0]))

# if __name__ == '__main__':
#     app.debug = True
#     app.run(port=8080)


if __name__ == "__main__":
    app.run(debug=True)
