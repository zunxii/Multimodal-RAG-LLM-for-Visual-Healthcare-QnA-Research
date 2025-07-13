from flask import Flask, request, jsonify
from utils.image_utils import get_image_embedding
from utils.query_utils import get_query_embedding, generate_summary
from utils.vector_db import search_similar_vectors
from utils.llm_utils import generate_final_answer

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    image = request.files.get('image')
    query = request.form.get('query')

    img_embedding = get_image_embedding(image)
    similar_vectors = search_similar_vectors(img_embedding)

    context_summary = generate_summary(similar_vectors)
    answer = generate_final_answer(query, context_summary)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
