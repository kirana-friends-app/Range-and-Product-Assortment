import os
import pandas as pd
from flask import Flask, request, send_from_directory, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_most_similar_product(new_description, df, description_column):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[description_column])
    new_description_vector = tfidf.transform([new_description])
    cosine_similarities = cosine_similarity(new_description_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    return df.iloc[most_similar_index]

@app.route('/')
def upload_form():
    return render_template('uploads.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Im here')
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the existing dataset
        existing_df = pd.read_excel(filepath, header=0)
        existing_df['ITEM TYPE'] = existing_df['ITEM TYPE'].astype(str)
        
        similar_products_list = []
        for index, row in existing_df.iterrows():
            item_name = str(row['ITEM TYPE'])  # Convert item_name to string
            most_similar_product = find_most_similar_product(item_name, existing_df, 'ITEM TYPE')
            similar_products_list.append(most_similar_product)

        predictions_df = pd.concat(similar_products_list, ignore_index=True)
        predictions_df['Original Product Description'] = existing_df['ITEM TYPE'].values

        print(predictions_df.head())
        
        output_filename = 'Predicted_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        predictions_df.to_excel(output_filepath, index=False)

        return send_from_directory(directory=UPLOAD_FOLDER, filename=output_filename)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
