from flask import Flask, request, redirect, url_for, render_template
import os
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, flash, send_file
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from flask import session
from uuid import uuid4
import numpy as np

app = Flask(__name__)
app.secret_key = 'Kirana@1234' 

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def find_most_similar_product(new_description, df, description_column):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[description_column])
    new_description_vector = tfidf.transform([new_description])
    cosine_similarities = cosine_similarity(new_description_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    most_similar_product = df.iloc[most_similar_index].copy()
    print('most_similar_product: ',most_similar_product)
    # Adjust this line if you want to filter different columns
    most_similar_product = most_similar_product[['CATEGORY', 'PRODUCT', 'BRAND', 'CATEGORY TYPE']]
    return most_similar_product.to_frame().T

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    session.clear()
    if request.method == 'POST':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mandatory_file = request.files['mandatoryExcel']
        optional_file = request.files.get('optionalExcel')  # This is optional

        # Initialize filenames to None or empty string before the conditions
        mandatory_filename = None
        optional_filename = None
        mandatory_columns = []
        optional_columns = []
        
        if mandatory_file and mandatory_file.filename:
            mandatory_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{mandatory_file.filename}")
            print('mandatory_filename: ',mandatory_filename)
            mandatory_file.save(mandatory_filename)
            session['mandatory_filename'] = mandatory_filename
            df_mandatory = pd.read_excel(mandatory_filename)
            mandatory_columns = df_mandatory.columns.tolist()
            
        if optional_file and optional_file.filename:
            optional_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{optional_file.filename}")
            optional_file.save(optional_filename)
            session['optional_filename'] = optional_filename
            df_optional = pd.read_excel(optional_filename)
            optional_columns = df_optional.columns.tolist()
            print()
            print()
            print('optional_file: ',optional_file)
            print()
            print()

        return render_template('upload_success.html', 
                               mandatory_filename=mandatory_file.filename if mandatory_file and mandatory_file.filename else None, 
                               optional_filename=optional_file.filename if optional_file and optional_file.filename else None,
                               mandatory_columns=mandatory_columns,
                               optional_columns=optional_columns)

    return render_template('index.html')


# @app.route('/process-column', methods=['POST'])
@app.route('/process-column', methods=['POST'])
def process_column():
    print("Session contents:")

    for key in session:
        print(f"{key}: {session[key]}")
    existing_catalog_path = os.path.join(app.config['UPLOAD_FOLDER'], "Merger (1).xlsx")
    existing_df = pd.read_excel(existing_catalog_path)
    # Retrieve file names and column names
    mandatory_filename = session.get('mandatory_filename')
    optional_filename = session.get('optional_filename', None)


    mandatory_product_description_column = request.form.get('mandatory_product_description')
    mandatory_quantity_sold_column = request.form.get('mandatory_quantity_sold')
    mandatory_price_column = request.form.get('mandatory_price')
    mandatory_category_column = request.form.get('mandatory_category')  # New

    optional_product_description_column = request.form.get('optional_product_description')
    optional_quantity_sold_column = request.form.get('optional_quantity_sold')
    optional_price_column = request.form.get('optional_price')
    optional_category_column = request.form.get('optional_category')  # New

    # Initialize DataFrame for combined data
    combined_df = pd.DataFrame(columns=[
        'Item Name', 'User Category', 'Category Predicted',  'Product Predicted',
        'Brand Predicted', 'Category Type', 'Quantity Sold First Excel', 
        'Price First Excel', 'Quantity Sold Second Excel', 
        'Price Second Excel', 'Quantity Growth', 'Value Growth'
    ])

    # Process mandatory file
    if mandatory_filename:
        mandatory_df = pd.read_excel(mandatory_filename)
        # Group by item description and sum the quantities and prices
        mandatory_agg = mandatory_df.groupby(mandatory_product_description_column).agg({
            mandatory_quantity_sold_column: 'sum',
            mandatory_price_column: 'sum'
        }).reset_index()

        mandatory_merged = mandatory_agg.merge(mandatory_df[[mandatory_product_description_column, mandatory_category_column]].drop_duplicates(),
                                           on=mandatory_product_description_column,
                                           how='left')


        # Check if the mandatory_category_column exists in mandatory_df
        # if mandatory_category_column not in mandatory_merged.columns:
        #     flash('Error: The category column specified does not exist in the mandatory file.', 'error')
        #     return redirect(url_for('upload_file'))  # Adjust this as necessary
        print(mandatory_agg.head())
        for index, row in mandatory_merged.iterrows():
            new_description = row[mandatory_product_description_column]
            userCat = row[mandatory_category_column]
            most_similar_product = find_most_similar_product(new_description, existing_df, 'Product Description')
            combined_df = combined_df.append({
                'Item Name': new_description,
                'User Category': userCat,
                'Category Predicted': most_similar_product['CATEGORY'].iloc[0],
                'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0],
                'Product Predicted': most_similar_product['PRODUCT'].iloc[0],
                'Brand Predicted': most_similar_product['BRAND'].iloc[0],
                'Quantity Sold First Excel': row[mandatory_quantity_sold_column],
                'Price First Excel': row[mandatory_price_column],
                'Quantity Sold Second Excel': None,
                'Price Second Excel': None
            }, ignore_index=True)

    # Process optional file
    

    if optional_filename:
        optional_df = pd.read_excel(optional_filename)
        # Group by item description and sum the quantities and prices
        optional_agg = optional_df.groupby(optional_product_description_column).agg({
            optional_quantity_sold_column: 'sum',
            optional_price_column: 'sum'
        }).reset_index()

        mandatory_merged1 = optional_agg.merge(optional_df[[optional_product_description_column, optional_category_column]].drop_duplicates(),
                                           on=optional_product_description_column,
                                           how='left')
        
        for index, row in mandatory_merged1.iterrows():
            new_description = row[optional_product_description_column]
            userCat = row[optional_category_column]
            # Check if this item is already in the combined_df
            if new_description in combined_df['Item Name'].values:
                combined_df.loc[combined_df['Item Name'] == new_description, 'Quantity Sold Second Excel'] = row[optional_quantity_sold_column]
                combined_df.loc[combined_df['Item Name'] == new_description, 'Price Second Excel'] = row[optional_price_column]
            else:
                most_similar_product = find_most_similar_product(new_description, existing_df, 'Product Description')
                combined_df = combined_df.append({
                    'Item Name': new_description,
                    'User Category': userCat,
                    'Category Predicted': most_similar_product['CATEGORY'].iloc[0] if not most_similar_product.empty else 'Unknown',
                    'Product Predicted': most_similar_product['PRODUCT'].iloc[0] if not most_similar_product.empty else 'Unknown',
                    'Brand Predicted': most_similar_product['BRAND'].iloc[0] if not most_similar_product.empty else 'Unknown',
                    'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0] if not most_similar_product.empty else 'Unknown',
                    'Quantity Sold First Excel': None,
                    'Price First Excel': None,
                    'Quantity Sold Second Excel': row[optional_quantity_sold_column],
                    'Price Second Excel': row[optional_price_column]
                }, ignore_index=True)
            
        total_price_second_excel = combined_df['Price Second Excel'].sum()
        category_price_sums_second = combined_df.groupby('User Category')['Price Second Excel'].sum()

        category_percentage_second = {}

        for category in category_price_sums_second.index:
            if total_price_second_excel > 0:  # Ensure there is no division by zero
                category_percentage_second[category] = (category_price_sums_second[category] / total_price_second_excel) * 100

        print('Last to Last Month Analysis')
        for category, percentage in category_percentage_second.items():
            print(f"Category optional'{category}' makes up {percentage:.2f}% of the Last to Last Month total price.")

        # Replace 0 with NaN to avoid division by zero error
        combined_df['Quantity Sold Second Excel'] = combined_df['Quantity Sold Second Excel'].replace(0, np.nan)

        # Now perform the division; any division by NaN will result in NaN instead of an error
        combined_df['Price ASP Second'] = combined_df['Price Second Excel'] / combined_df['Quantity Sold Second Excel']

        # If you want to handle NaN values in 'Price ASP Second', you can fill them with a default value
        # For example, filling NaN with 0 or any other placeholder value
        combined_df['Price ASP Second'] = combined_df['Price ASP Second'].fillna(0)
    


    # Calculate Growth
    for index, row in combined_df.iterrows():
        if pd.notna(row['Quantity Sold First Excel']) and pd.notna(row['Quantity Sold Second Excel']):
            combined_df.at[index, 'Quantity Growth'] = "{:.0%}".format((row['Quantity Sold First Excel'] - row['Quantity Sold Second Excel']) / row['Quantity Sold Second Excel'])
        if pd.notna(row['Price First Excel']) and pd.notna(row['Price Second Excel']):
            combined_df.at[index, 'Value Growth'] = "{:.0%}".format((row['Price First Excel'] - row['Price Second Excel']) / row['Price Second Excel'])
        
    
    # Save combined data to a new Excel file
    # Generate a unique identifier for the filename
    unique_id = uuid4()
    output_filename = f"StructuredData_{unique_id}.xlsx"
    output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    combined_df.to_excel(output_file_path, index=False)

    #Logic fisrt
    total_price_first_excel = combined_df['Price First Excel'].sum()

    category_price_sums_first = combined_df.groupby('User Category')['Price First Excel'].sum()

    # Create a dictionary to store the percentage of each category type
    category_percentage_first = {}

    # Calculate the percentage for each category in first and second excel
    for category in category_price_sums_first.index:
        category_percentage_first[category] = (category_price_sums_first[category] / total_price_first_excel) * 100

    print('Last Month Analysis')
    for category, percentage in category_percentage_first.items():
        print(f"Category mandatory'{category}' makes up {percentage:.2f}% of the Last Month total price.")

    if mandatory_filename and optional_filename:
        # Assuming 'Quantity Growth' already exists in 'combined_df'
        # and is formatted correctly for sorting (numeric values).

        # Convert 'Quantity Growth' to numeric if it's not already, handling any potential conversion errors
        combined_df['Quantity Growth'] = pd.to_numeric(combined_df['Quantity Growth'].str.rstrip('%'), errors='coerce')
        # Sort the DataFrame by 'Quantity Growth' in descending order to get the highest growth products at the top
        sorted_combined_df = combined_df.sort_values(by='Quantity Growth', ascending=False)
        # Now, select the top 50 unique products based on 'Quantity Growth'
        top_50_products = sorted_combined_df.drop_duplicates(subset=['Product Predicted'], keep='first').head(50)       
        # Save the top 50 products to a new Excel file for download or display
        top_50_filename = f"Top50Products_{uuid4()}.xlsx"
        top_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], top_50_filename)
        top_50_products.to_excel(top_50_file_path, index=False)


        sorted_combined_df_degrowth = combined_df.sort_values(by='Quantity Growth', ascending=True)
        # Select the bottom 50 unique products based on 'Quantity Growth'
        bottom_50_products_degrowth = sorted_combined_df_degrowth.drop_duplicates(subset=['Product Predicted'], keep='first').head(50)
        # Save the bottom 50 products to a new Excel file for download or display
        bottom_50_filename = f"Bottom50Products_{uuid4()}.xlsx"
        bottom_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], bottom_50_filename)
        bottom_50_products_degrowth.to_excel(bottom_50_file_path, index=False)



        combined_df['Total Sales'] = combined_df['Quantity Sold First Excel'] * combined_df['Price First Excel']       
        # Step 2: Sort SKUs by Total Sales in descending order
        sorted_df = combined_df.sort_values(by='Total Sales', ascending=False)       
        # Step 3: Calculate Cumulative Sales Percentage
        sorted_df['Cumulative Sales'] = sorted_df['Total Sales'].cumsum()
        total_sales = sorted_df['Total Sales'].sum()
        sorted_df['Cumulative Sales Percentage'] = 100 * sorted_df['Cumulative Sales'] / total_sales       
        # Step 4: Identify SKUs contributing to 80% of sales
        cutoff_index = sorted_df['Cumulative Sales Percentage'].searchsorted(80)
        top_skus = sorted_df.iloc[:cutoff_index+1]       
        # Step 5: Calculate Category Percentages within the top SKUs
        category_sales = top_skus.groupby('User Category')['Total Sales'].sum()
        category_percentage = 100 * category_sales / top_skus['Total Sales'].sum()
        # Convert Category Percentages to DataFrame for easier handling/display
        category_percentage_df = category_percentage.reset_index(name='Category Percentage')
        # Save or display your results as needed
        # For example, saving the top SKUs and category percentages to Excel files
        top_skus_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'TopSKUs_Contributing_80_Percent_Sales.xlsx')
        top_skus.to_excel(top_skus_filename, index=False)
        category_percentage_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Category_Percentages_of_Top_SKUs.xlsx')
        category_percentage_df.to_excel(category_percentage_filename, index=False)

        
        flash('Top 50 high selling products list is ready.')
        # Assuming you want to offer this file for download
        return send_file(top_50_file_path, as_attachment=True, download_name=top_50_filename)

    

    flash('Processing complete. Structured data saved.')
    session.clear()
    return send_file(output_file_path, as_attachment=True, download_name=output_filename)

if __name__ == '__main__':
    app.run(debug=True)
