from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

def calculate_target_sku(store_size):
    sku_ranges = [
        (0, 200, 3), (200, 300, 2.9), (300, 400, 2.8), (400, 500, 2.7),
        (500, 600, 2.6), (600, 700, 2.5), (700, 800, 2.4), (800, 900, 2.3),
        (900, 1000, 2.2), (1000, 1100, 2.1), (1100, 1200, 2)
    ]
    for lower_bound, upper_bound, sku_psf in sku_ranges:
        if lower_bound < store_size <= upper_bound:
            return store_size * sku_psf
    return store_size * 2  # Default case for store_size > 1200

# Your category percentages
category_percentages = {
    # Your category percentages here
    "Masala And Spices": 6,
    "Dryfruits": 4,
    "Farshan/Chikki/Mithai": 2,
    "Upwas Special": 1,
    "Grocery": 22,
    "Ready To Eat/Instant Food": 2,
    "Papad/Pickles": 1,
    "Baby Foods And Health Foods/ Drinks": 2,
    "Biscuits/Bakery Products": 4,
    "Jams/Ketchups/Spreads": 1,
    "Chocolates And Confectionaries": 1,
    "Dairy Products": 1,
    "Frozen Foods": 1,
    "Cold Drink/Juices/Drinks": 2,
    "Tea/Coffee": 3,
    "Ghee/Vanaspati/Oils": 14,
    "Sugar/Salt": 3,
    "Pooja Items": 1,
    "Paper/Party Products": 1,
    "Cleaning Material": 4,
    "Cosmetics": 2,
    "Sanitary Napkins & Diapers": 1,
    "Dental Care": 3,
    "Mosquito Repellents And Pesticides": 1,
    "Shaving Care Products": 0.6,
    "Deo/Talcum/Perfume": 1,
    "Detergent/Washing Aids": 6,
    "Hair Care": 3,
    "Others": 0.8,
    "Bath Soap/Liquids/Sanitizers": 5,
    "Shoe Care Products":0.6
}

input_file_path = "Merger.csv"
data = pd.read_csv(input_file_path)
data.columns = data.iloc[0]
data = data[1:]

def distribute_skus_corrected(data, category_sku_counts):
    selected_rows = pd.DataFrame()  # This will now be a single DataFrame
    for category, sku_count in category_sku_counts.items():
        category_data = data[(data['CATEGORY'] == category) & (data['Product Must Have'] == 'Yes')]
        sorted_category_data = category_data.sort_values(by='PRODUCT PRIORITIES')
        selected_products = []
        cumulative_products = 0
        for priority, group in sorted_category_data.groupby('PRODUCT PRIORITIES'):
            if cumulative_products + len(group) <= sku_count:
                selected_products.extend(group.index.tolist())
                cumulative_products += len(group)
            else:
                needed = sku_count - cumulative_products
                selected_products.extend(group.head(needed).index.tolist())
                break
        # Add the category column for each selected product
        category_selection = data.loc[selected_products]
        category_selection['Category'] = category  # Add category name as a new column
        selected_rows = pd.concat([selected_rows, category_selection])  # Concatenate with the overall DataFrame
    return selected_rows

@app.route('/calculate', methods=['POST'])
def calculate_skus():
    try:
        request_data = request.get_json()
        store_size = float(request_data['storeSize'])
        
        total_skus = calculate_target_sku(store_size)
        category_sku_counts = {cat: int((perc / 100) * total_skus) for cat, perc in category_percentages.items()}
        selected_rows = distribute_skus_corrected(data, category_sku_counts)
        
        filtered_selected_rows = selected_rows[['CATEGORY', 'PRODUCT', 'BRAND', 'Product Description']]
        
        # Convert the filtered_selected_rows DataFrame into a list of dictionaries
        rows_list = filtered_selected_rows.to_dict(orient='records')

        response_data = {
            'totalSKUs': total_skus,
            'totalProductCount': len(filtered_selected_rows),
            'uniqueCategoryCount': filtered_selected_rows['CATEGORY'].nunique(),
            'products': rows_list  # Add this line to include the product data
        }

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)