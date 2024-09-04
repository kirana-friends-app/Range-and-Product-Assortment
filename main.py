from flask import jsonify
from flask import jsonify
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, render_template
from flask import Flask, request, redirect, url_for, render_template, flash, send_file, session
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from uuid import uuid4
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from flask import session
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER


app = Flask(__name__)
app.secret_key = 'Kirana@1234' 

UPLOAD_FOLDER = '/home/ubuntu/SalesAnalysisTool/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def add_table_to_story(dataframe, title, story, styles):
    story.append(Paragraph(title, styles['Heading2']))
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

def save_pie_chart(data, filename, title):
    labels = data.keys()
    sizes = data.values()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def find_most_similar_product(new_description, df, description_column):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[description_column])
    new_description_vector = tfidf.transform([new_description])
    cosine_similarities = cosine_similarity(new_description_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    most_similar_product = df.iloc[most_similar_index].copy()
    # Adjust this line if you want to filter different columns
    most_similar_product = most_similar_product[['CATEGORY', 'PRODUCT', 'BRAND', 'CATEGORY TYPE']]
    # Extract the highest similarity score (i.e., the match percentage)
    highest_similarity_score = cosine_similarities.max()
    # print("new_description: ",new_description)
    # print("highest_similarity_score: ",highest_similarity_score)
    # print("most_similar_product: ",most_similar_product)
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
            # print('mandatory_filename: ',mandatory_filename)
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
            # print()
            # print()
            # print('optional_file: ',optional_file)
            # print()
            # print()
        # if mandatory_file
        #     return mandatory_columns
        # else:
        #     return optional_columns
        return render_template('upload_success.html', 
                               mandatory_filename=mandatory_file.filename if mandatory_file and mandatory_file.filename else None, 
                               optional_filename=optional_file.filename if optional_file and optional_file.filename else None,
                               mandatory_columns=mandatory_columns,
                               optional_columns=optional_columns)

    return render_template('index.html')


@app.route('/process-one', methods=['POST'])
def process_one():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_catalog_path = os.path.join(app.config['UPLOAD_FOLDER'], "Merger (1).xlsx")
    existing_df = pd.read_excel(existing_catalog_path)

    # Retrieve file names and column names
    # mandatory_filename = request.form.get('mandatory_filename')
    if 'mandatory_filename' not in request.files:
        return "No file part", 400
    
    mandatory_filename = request.files['mandatory_filename']
    if mandatory_filename.filename == '':
        return "No selected file", 400
    
    if mandatory_filename:
        filename = secure_filename(mandatory_filename.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        mandatory_filename.save(file_path)
        
        # Read the uploaded file
        mandatory_df = pd.read_excel(file_path)

        mandatory_product_description_column = request.form.get('mandatory_product_description')
        mandatory_quantity_sold_column = request.form.get('mandatory_quantity_sold')
        mandatory_price_column = request.form.get('mandatory_price')
        mandatory_category_column = request.form.get('mandatory_category', None)

        # Retrieve number of bills
        num_of_bills = request.form.get('num_of_bills', default=1, type=int)
        store_szie = request.form.get('store_size', default=1, type=int)

        # Initialize DataFrame for combined data
        combined_df = pd.DataFrame(columns=[
            'Item Name', 'User Category', 'Category Predicted',  'Product Predicted',
            'Brand Predicted', 'Category Type', 'Quantity Sold First Excel', 
            'Price First Excel', 'Quantity Sold Second Excel', 
            'Price Second Excel', 'Quantity Growth', 'Value Growth'
        ])

        # Process mandatory file
        # if mandatory_filename:
        #mandatory_df = pd.read_excel(mandatory_filename)
        # Group by item description and sum the quantities and prices
        mandatory_agg = mandatory_df.groupby(mandatory_product_description_column).agg({
            mandatory_quantity_sold_column: 'sum',
            mandatory_price_column: 'sum'
        }).reset_index()

        mandatory_merged = mandatory_agg.merge(mandatory_df[[mandatory_product_description_column, mandatory_category_column]].drop_duplicates(),
                                            on=mandatory_product_description_column,
                                            how='left')
        new_rows = []
        for index, row in mandatory_merged.iterrows():
            new_description = row[mandatory_product_description_column]
            userCat = row[mandatory_category_column]
            most_similar_product = find_most_similar_product(new_description, existing_df, 'Product Description')
            
            new_row = {
                'Item Name': row[mandatory_product_description_column],
                'User Category': row[mandatory_category_column],
                'Category Predicted': most_similar_product['CATEGORY'].iloc[0],
                'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0],
                'Product Predicted': most_similar_product['PRODUCT'].iloc[0],
                'Brand Predicted': most_similar_product['BRAND'].iloc[0],
                'Quantity Sold First Excel': row[mandatory_quantity_sold_column],
                'Price First Excel': row[mandatory_price_column],
                'Quantity Sold Second Excel': None,
                'Price Second Excel': None
            }
            new_rows.append(new_row)

        new_rows_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)

        total_price_first_excel = combined_df['Price First Excel'].sum()
        total_item_sold = combined_df['Quantity Sold First Excel'].sum()
        category_price_sums_first = combined_df.groupby('Category Type')['Price First Excel'].sum()
        category_percentage_first = {}

        # Calculate the percentage for each category in first and second excel
        for category in category_price_sums_first.index:
            category_percentage_first[category] = (category_price_sums_first[category] / total_price_first_excel) * 100
        print('Last Month Analysis')
        for category, percentage in category_percentage_first.items():
            print(f"Category mandatory'{category}' makes up {percentage:.2f}% of the Last Month total price.")
        print()
        combined_df['Quantity Sold First Excel'] = combined_df['Quantity Sold First Excel'].replace(0, np.nan)
        combined_df['Price ASP First'] = combined_df['Price First Excel'] / combined_df['Quantity Sold First Excel']
        combined_df['Price ASP First'] = combined_df['Price ASP First'].fillna(0)

        # to print the category of the customer
        category_price_sums_first1 = combined_df.groupby('User Category')['Price First Excel'].sum()
        category_percentage_first1 = {}
        for category in category_price_sums_first1.index:
            category_percentage_first1[category] = (category_price_sums_first1[category] / total_price_first_excel) * 100
        print('Last Month Analysis')
        for category, percentage in category_percentage_first1.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price.")
        
        # Sort the dictionary by percentage in descending order to get the top 5
        sorted_categories_top5 = sorted(category_percentage_first1.items(), key=lambda x: x[1], reverse=True)[:5]

        # Sort the dictionary by percentage in ascending order to get the bottom 5
        sorted_categories_bottom5 = sorted(category_percentage_first1.items(), key=lambda x: x[1])[:5]

        ABV = total_price_first_excel/num_of_bills
        print('Average Basket Value: ',ABV)
        print()
        ABS = total_item_sold/num_of_bills
        print("Average Basket Sales: ",ABS)
        print()
        AIV = total_price_first_excel/total_item_sold
        print("Average Item Value: ",AIV)
        print()
        sqft = total_price_first_excel/store_szie
        print("Average Item Value: ",sqft)
        print()

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


        # Destination Food % for One Month
        # print("Destination Food for Last Month")
        destination_food_percentage = category_percentage_first.get('Destination Food', 0)
        destination_food_message = ""
        destination_insights = ""
        if destination_food_percentage > 45:
            destination_food_message = f""
        elif destination_food_percentage <= 45:
            destination_insights = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. "
            destination_food_message = (
        f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%."
        " Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers."
        " You can do it by:\n"
        "Enhancing your product range\n"
        "Ensuring good display\n"
        "Ensuring product availability\n"
        "Planning promotions"
    )
        print(destination_food_message)
        print()
        
        if destination_food_percentage <= 45:
            destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
            total_price_first_excel1 = combined_df['Price First Excel'].sum()
            total_item_sold = combined_df['Quantity Sold First Excel'].sum()
            category_price_sums_first10 = destination_food_df.groupby('Category Type')['Price First Excel'].sum()
            filename = f"Destination_Food_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            destination_food_df.to_excel(filepath, index=False)
            
            destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
            total_price_destination = combined_df['Price First Excel'].sum()
            category_price_sums_destination = destination_food_df.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destinationNew = {category: (price / total_price_destination) * 100 for category, price in category_price_sums_destination.items()}
            for category, percentage in category_percentage_destinationNew.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destinationNew.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destinationNew.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destinationNew = {}
        
        session['destinationInformation'] = {"destination_food_percentage": destination_food_percentage, "destination_food_message": destination_food_message, "destination_insights": destination_insights, "category_percentage_destinationNew": category_percentage_destinationNew}

        
        # Calculate the combined percentage of Destination Food and Routine Non Core Food
        print("All Food Categories for Last Month")
        combined_percentage = category_percentage_first.get('Destination Food', 0) + category_percentage_first.get('Routine Non Core Food', 0)
        substantially_below = "substantially " if combined_percentage < 50 else ""
        food_categories_message = ""
        food_categories_insights = ""

        if combined_percentage > 65:
            food_categories_message = f""
        else:
            food_categories_insights = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
            food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall {substantially_below}the desired level and is at {combined_percentage:.2f}%. Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        # print("food categories message message for Last month")
        print(food_categories_message)
        print()

        if combined_percentage < 65:
            food_categories_df = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
            # Generate a unique timestamp to append to the filename
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_categories_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            food_categories_df.to_excel(filepath, index=False)

            food_categories_df100 = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
            total_price_destination100 = combined_df['Price First Excel'].sum()
            category_price_sums_destination100 = food_categories_df100.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destination100 = {category: (price / total_price_destination100) * 100 for category, price in category_price_sums_destination100.items()}
            for category, percentage in category_percentage_destination100.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destination100.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination100.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destination100 = {}

        session['AllFoodCategories'] = {"combined_percentage": combined_percentage, "food_categories_message": food_categories_message, "food_categories_insights": food_categories_insights, "category_percentage_destination100": category_percentage_destination100}


        print("Non Food Categories for Last Month")
        routine_non_food_percentage = category_percentage_first.get('Routine Non Food', 0)
        routine_non_food_message = ""
        routine_non_food_insights = ""

        if routine_non_food_percentage < 30:
            routine_non_food_insights = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
            routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        else:
            routine_non_food_message = f""
        # print("routine non food message message message for Last month")
        print(routine_non_food_message)
        print()
        if routine_non_food_percentage < 30:
            Routine_Non_Food_df = combined_df[combined_df['Category Type'] == 'Routine Non Food']
            # Generate a unique timestamp to append to the filename
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Routine_Non_Food_Items_{timestamp}.xlsx"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Routine_Non_Food_df.to_excel(filepath, index=False)

            Routine_Non_Food_df101 = combined_df[combined_df['Category Type'] == 'Routine Non Food']
            total_price_destination101 = combined_df['Price First Excel'].sum()
            category_price_sums_destination101 = Routine_Non_Food_df101.groupby('User Category')['Price First Excel'].sum()
            category_percentage_destination101 = {category: (price / total_price_destination101) * 100 for category, price in category_price_sums_destination101.items()}
            for category, percentage in category_percentage_destination101.items():
                print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
            data = {
                "User Category": list(category_percentage_destination101.keys()),
                "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination101.values()]
            }
            df = pd.DataFrame(data)
        else:
            category_percentage_destination101 = {}

        session['NonFoodCategories'] = {"routine_non_food_percentage": routine_non_food_percentage, "routine_non_food_message": routine_non_food_message, "routine_non_food_insights": routine_non_food_insights, "category_percentage_destination101": category_percentage_destination101}

        # Salt Circle Analysis
        # Step 1: Filter combined_df for Sugar/Salt category and Salt product
        salt_circle_df = combined_df[(combined_df['Category Predicted'] == 'Sugar/Salt') & (combined_df['Product Predicted'] == 'Salt')]
        # Step 2: Calculate No_of_Salt_Packets
        No_of_Salt_Packets = salt_circle_df['Quantity Sold First Excel'].sum()
        # Check to avoid division by zero
        if No_of_Salt_Packets > 0:
            # Step 3: Calculate No of Families Shopping Estimate
            Total_Sales = combined_df['Price First Excel'].sum()
            No_of_Families_Shopping_Estimate = Total_Sales / No_of_Salt_Packets
        else:
            No_of_Families_Shopping_Estimate = 0
        # Fixed value for Average Monthly Basket Estimate
        Average_Monthly_Basket_Estimate = 4000  # Rs
        # Step 4: Compare and determine if it's greater or less
        if No_of_Families_Shopping_Estimate > Average_Monthly_Basket_Estimate:
            salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
            comparison_message = f"The No of Families Shopping Estimate is greater than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate:.2f}"
        elif No_of_Families_Shopping_Estimate < Average_Monthly_Basket_Estimate:
            salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
            comparison_message = f"The No of Families Shopping Estimate is less than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate}"
        else:
            salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
        print(comparison_message)

        # Top SKU Contributing to 80% of Sales
        if mandatory_filename:
            total_sku = len(combined_df)
            total_sales = combined_df['Price First Excel'].sum()
            df_sorted = combined_df.sort_values(by='Price First Excel', ascending=False)
            df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
            df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
            cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
            top_80_percent_itemsSKU = df_sorted.loc[:cutoff_index]
            num_top_80_percent_items = len(top_80_percent_itemsSKU)
            percentageofsku = (num_top_80_percent_items / total_sku) * 100
            SKU_msg = f"Total SKU is {total_sku}. Total Sales is Rs{total_sales:.2f}. {num_top_80_percent_items} SKU's contribute to 80% of sales which is {percentageofsku:.2f}% of total SKU's."
            # output_path = 'Top_80_Percent_High_Selling_Sku.xlsx'
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'SKU_Contributing_80%_sales_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            top_80_percent_itemsSKU.to_excel(output_path, index=False)
            top_20_itemsSKU = top_80_percent_itemsSKU.head(20)[['Item Name', 'Price First Excel']]
            top_20_itemsSKU = top_20_itemsSKU.rename(columns={
                'Price First Excel': 'Price Last Month',
            })
            data103 = [top_20_itemsSKU.columns.tolist()] + top_20_itemsSKU.values.tolist()
            # Get the bottom 20 entries
            bottom_20 = df_sorted.tail(20)[['Item Name', 'Price First Excel']]
            bottom_20_itemsSKU = bottom_20.rename(columns={
                'Price First Excel': 'Price Last Month',
            })
            data199 = [bottom_20_itemsSKU.columns.tolist()] + bottom_20_itemsSKU.values.tolist()




        # Top Product Contributing to 80% of Sales
        if mandatory_filename:
            aggregated_data = combined_df.groupby('Product Predicted').agg({
                'Quantity Sold First Excel': 'sum',
                'Price First Excel': 'sum',
            }).reset_index()
            total_sales = aggregated_data['Price First Excel'].sum()
            df_sorted = aggregated_data.sort_values(by='Price First Excel', ascending=False)
            df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
            df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
            cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
            top_80_percent_items = df_sorted.loc[:cutoff_index]
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'Product_Contributing_80%_sales_{timestamp}.xlsx'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
            top_80_percent_items.to_excel(output_path, index=False)
            top_20_itemsproducts = top_80_percent_items.head(20)[['Product Predicted', 'Price First Excel']]
            top_20_itemsproducts = top_20_itemsproducts.rename(columns={
                'Price First Excel': 'Price Last Month',
            })
            data104 = [top_20_itemsproducts.columns.tolist()] + top_20_itemsproducts.values.tolist()


        # Pie Chart Generation
        save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')


        # Pdf Generation
        pdf_filename = f'Sales_Analysis_Report_{timestamp}.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Section : Purpose
        styles.add(ParagraphStyle(name='CenteredBoldHeading2', parent=styles['Heading2'], alignment=TA_CENTER, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

        # Header Section
        story.append(Spacer(1, 12))
        story.append(Paragraph("Sales Analysis Insight & Action Report", styles['CenteredBoldHeading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("A sales analysis tool is a software application that helps businesses track, analyze, and visualize their sales data to gain insights into sales performance. It typically offers features like sales forecasting, trend analysis, and performance metrics. These tools enable companies to make data-driven decisions, identify sales opportunities, optimize sales strategies, and enhance customer satisfaction. They are commonly used by sales teams, managers, and executives to monitor sales activities and achieve business goals.", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 1: Key Performance Indicators
        story.append(Paragraph("KEY Performance Indicators", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Basket Value (ABV): Average Value of each bill", styles['BodyText']))
        story.append(Paragraph(f"(a) Average Basket Value (ABV) ={ABV:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Basket Sales (ABS): Average number of items in each bill", styles['BodyText']))
        story.append(Paragraph(f"(b) Average Basket Sales (ABS) = {ABS:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Item Value (ASP): Average price of each item sold", styles['BodyText']))
        story.append(Paragraph(f"(c) Average Item Value (ASP) = {AIV:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(d) Per Sq. ft.sales = {sqft:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(e) Number of bills: {num_of_bills}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(f) Current Month sale Value: {total_price_first_excel:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(g) Store Size: {store_szie}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))

        story.append(Paragraph(" Family Basket", styles['Heading1']))
        story.append(Paragraph("Salt Circle", styles['BodyText']))
        story.append(Paragraph("This part of the report shows you the estimates of numbers of families who have shopped with you in the given month and their shopping basket assessment.", styles['BodyText']))
        story.append(Paragraph(f"{salt_msg}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        # Section 2: Business Group Performance
        story.append(Paragraph("Business Group Performance", styles['Heading1']))
        story.append(Paragraph("This part of the analysis shows you the performance of your various business groups that consist of similar categories in the way customers see them. It considers the benchmark participation of these business groups and provides you insights and actions required,", styles['BodyText']))
        last_month_image = Image('last_month_pie_chart.png')
        last_month_image._restrictSize(400, 400)
        story.append(last_month_image)
        if destination_food_message.strip():
            story.append(Paragraph("Destination Category %", styles['BodyText']))
            destination_insights1 = Paragraph(destination_insights, styles['BodyText'])
            story.append(destination_insights1)
            story.append(Spacer(1, 12))
            destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
            story.append(destination_food_paragraph)
            columns_to_display = [
                "Item Name",
                "Quantity Sold First Excel",
                "Price First Excel",
            ]
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destinationNew.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if food_categories_message.strip():
            story.append(Paragraph("All Food Category %", styles['BodyText']))
            line12 = Paragraph(food_categories_insights, styles['BodyText'])
            story.append(line12)
            story.append(Spacer(1, 12))
            line = Paragraph(food_categories_message, styles['BodyText'])
            story.append(line)
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination100.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if routine_non_food_message.strip():
            story.append(Paragraph("Non Food Category %", styles['BodyText']))
            line13 = Paragraph(routine_non_food_insights, styles['BodyText'])
            story.append(line13)
            story.append(Spacer(1, 12))
            line1 = Paragraph(routine_non_food_message, styles['BodyText'])
            story.append(line1)
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination101.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        story.append(Paragraph("Top 5 Contributing Category %", styles['Heading1']))
        table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in sorted_categories_top5]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Bottom 5 Contributing Category %", styles['Heading1']))
        table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in sorted_categories_bottom5]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(PageBreak())
        # Section Top SKU
        story.append(Paragraph("Top 20 SKU", styles['Heading1']))
        story.append(Paragraph("This part of the report shows you business growth at SKU level", styles['BodyText']))
        table = Table(data103)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        # Section Top Product
        story.append(Paragraph("Top 20 Products", styles['Heading1']))
        story.append(Paragraph("This part of the report shows you business growth at Product level", styles['BodyText']))
        story.append(Spacer(1, 12))
        table = Table(data104)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph("Bottom 20 Products", styles['Heading1']))
        table = Table(data199)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        pdf.build(story)
        print(f"PDF generated: {pdf_filename}")
        flash('Processing complete. Structured data saved.')
        session['pdf_path'] = pdf_path

        data = {                                                                                                                                                "ABV": ABV,
            "ABS": ABS
                }

        return jsonify({"ABV": ABV})
    #sqft, ABV , ABS #, AIV ,num_of_bills,total_price_first_excel,No_of_Salt_Packets, No_of_Families_Shopping_Estimate, Total_Sales, category_percentage_first
        # return send_file(output_file_path, as_attachment=True, download_name=output_filename)

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    print("Session contents:", session)
    # Retrieve the path to the PDF from the session
    pdf_path = session.get('pdf_path')
    if not pdf_path:
        return "No PDF file available for download", 404

    # Send the PDF file to the user
    return  True
#send_file(pdf_path, as_attachment=True, download_name='Sales_Analysis_Report.pdf')



@app.route('/BusinessGroupPerformanceDetailed', methods=['POST'], endpoint='business_group_performance')
def BusinessGroupPerformanceDetailed():
    print("Session contents:")
    for key in session:
        print(f"{key}: {session[key]}")
    
    destination_food_percentage = session.get('destinationInformation').get('destination_food_percentage')
    destination_food_message = session.get('destinationInformation').get('destination_food_message')
    destination_insights = session.get('destinationInformation').get('destination_insights')
    category_percentage_destinationNew = session.get('destinationInformation').get('category_percentage_destinationNew')
    all_food_categories_percentage = session.get('AllFoodCategories').get('combined_percentage')
    food_categories_message = session.get('AllFoodCategories').get('food_categories_message')
    food_categories_insights = session.get('AllFoodCategories').get('food_categories_insights')
    category_percentage_destination100 = session.get('AllFoodCategories').get('category_percentage_destination100')
    routine_non_food_percentage = session.get('NonFoodCategories').get('routine_non_food_percentage')
    routine_non_food_message = session.get('NonFoodCategories').get('routine_non_food_message')
    routine_non_food_insights = session.get('NonFoodCategories').get('routine_non_food_insights')
    category_percentage_destination101 = session.get('NonFoodCategories').get('category_percentage_destination101')

    return True 
#destination_food_percentage, destination_food_message, destination_insights, category_percentage_destinationNew, all_food_categories_percentage, food_categories_message, food_categories_insights, category_percentage_destination100, routine_non_food_percentage, routine_non_food_message, routine_non_food_insights, category_percentage_destination101,category_percentage_first1






# @app.route('/process-column', methods=['POST'])
@app.route('/process-both', methods=['POST'])
def process_column1():
    # Here im just Printing what i have uploaded
    print("Session contents:")
    for key in session:
        print(f"{key}: {session[key]}")
    ##############################

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    existing_catalog_path = os.path.join(app.config['UPLOAD_FOLDER'], "Merger (1).xlsx")
    existing_df = pd.read_excel(existing_catalog_path)

    # Retrieve file names and column names
    mandatory_filename = session.get('mandatory_filename')
    optional_filename = session.get('optional_filename', None)


    mandatory_product_description_column = request.form.get('mandatory_product_description')
    mandatory_quantity_sold_column = request.form.get('mandatory_quantity_sold')
    mandatory_price_column = request.form.get('mandatory_price')
    mandatory_category_column = request.form.get('mandatory_category', None)  # New

    optional_product_description_column = request.form.get('optional_product_description')
    optional_quantity_sold_column = request.form.get('optional_quantity_sold')
    optional_price_column = request.form.get('optional_price')
    optional_category_column = request.form.get('optional_category', None)  # New

    # Retrieve number of bills
    num_of_bills = request.form.get('num_of_bills', default=1, type=int)
    no_of_bills  = request.form.get('num_of_billsPrevious', default=1, type=int)
    store_szie = request.form.get('store_size', default=1, type=int)

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
        new_rows = []
        for index, row in mandatory_merged.iterrows():
            new_description = row[mandatory_product_description_column]
            userCat = row[mandatory_category_column]
            most_similar_product = find_most_similar_product(new_description, existing_df, 'Product Description')
            
            new_row = {
                'Item Name': row[mandatory_product_description_column],
                'User Category': row[mandatory_category_column],
                'Category Predicted': most_similar_product['CATEGORY'].iloc[0],
                'Category Type': most_similar_product['CATEGORY TYPE'].iloc[0],
                'Product Predicted': most_similar_product['PRODUCT'].iloc[0],
                'Brand Predicted': most_similar_product['BRAND'].iloc[0],
                'Quantity Sold First Excel': row[mandatory_quantity_sold_column],
                'Price First Excel': row[mandatory_price_column],
                'Quantity Sold Second Excel': None,
                'Price Second Excel': None
            }
            new_rows.append(new_row)

        new_rows_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)

        total_price_first_excel = combined_df['Price First Excel'].sum()
        total_item_sold = combined_df['Quantity Sold First Excel'].sum()
        category_price_sums_first = combined_df.groupby('Category Type')['Price First Excel'].sum()
        category_percentage_first = {}

        # Calculate the percentage for each category in first and second excel
        for category in category_price_sums_first.index:
            category_percentage_first[category] = (category_price_sums_first[category] / total_price_first_excel) * 100

        print('Last Month Analysis')
        for category, percentage in category_percentage_first.items():
            print(f"Category mandatory'{category}' makes up {percentage:.2f}% of the Last Month total price.")
        print()
        # Replace 0 with NaN to avoid division by zero error
        combined_df['Quantity Sold First Excel'] = combined_df['Quantity Sold First Excel'].replace(0, np.nan)

        # Now perform the division; any division by NaN will result in NaN instead of an error
        combined_df['Price ASP First'] = combined_df['Price First Excel'] / combined_df['Quantity Sold First Excel']

        # If you want to handle NaN values in 'Price ASP Second', you can fill them with a default value
        # For example, filling NaN with 0 or any other placeholder value
        combined_df['Price ASP First'] = combined_df['Price ASP First'].fillna(0)

        # to print the category of the customer
        category_price_sums_first1 = combined_df.groupby('User Category')['Price First Excel'].sum()
        category_percentage_first1 = {}

        # Calculate the percentage for each category in first and second excel
        for category in category_price_sums_first1.index:
            category_percentage_first1[category] = (category_price_sums_first1[category] / total_price_first_excel) * 100

        print('Last Month Analysis')
        for category, percentage in category_percentage_first1.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price.")
        print()

        ABV = total_price_first_excel/no_of_bills
        print('Average Basket Value: ',ABV)
        print()

        ABS = total_item_sold/no_of_bills
        print("Average Basket Sales: ",ABS)
        print()

        AIV = total_price_first_excel/total_item_sold
        print("Average Item Value: ",AIV)
        print()

        sqft = total_price_first_excel/store_szie
        print("Average Item Value: ",sqft)
        print()

        GwothAPICuurent = (ABS*AIV*no_of_bills)

        category_price_sums_first12 = combined_df.groupby('User Category')['Quantity Sold First Excel'].sum()
        category_percentage_first12 = {}

        # Calculate the percentage for each category in first and second excel
        for category in category_price_sums_first12.index:
            category_percentage_first12[category] = (category_price_sums_first12[category] / total_item_sold) * 100

        
        table_data200 = [['User Category', 'Value%', 'Volume%']]
        for category in category_percentage_first1:
            value_percent = category_percentage_first1.get(category, 0)  # Get value percent, default to 0 if not found
            volume_percent = category_percentage_first12.get(category, 0)  # Get volume percent, default to 0 if not found
            table_data200.append([category, f"{value_percent:.2f}%", f"{volume_percent:.2f}%"])





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
        new_rows = []
        for index, row in mandatory_merged1.iterrows():
            new_description = row[optional_product_description_column]
            userCat = row[optional_category_column]
            # Check if this item is already in the combined_df
            if new_description in combined_df['Item Name'].values:
                combined_df.loc[combined_df['Item Name'] == new_description, 'Quantity Sold Second Excel'] = row[optional_quantity_sold_column]
                combined_df.loc[combined_df['Item Name'] == new_description, 'Price Second Excel'] = row[optional_price_column]
            else:
                most_similar_product = find_most_similar_product(new_description, existing_df, 'Product Description')
                new_row = {
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
                }
                new_rows.append(new_row)
        # Only update combined_df if there are new rows to add
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)  # Convert list of dicts to DataFrame
            combined_df = pd.concat([combined_df, new_rows_df], ignore_index=True)

        total_price_second_excel = combined_df['Price Second Excel'].sum()
        total_item_sold_second_excel = combined_df['Quantity Sold Second Excel'].sum()
        # category_price_sums_second = combined_df.groupby('User Category')['Price Second Excel'].sum()
        category_price_sums_second = combined_df.groupby('Category Type')['Price Second Excel'].sum()

        category_percentage_second = {}

        for category in category_price_sums_second.index:
            if total_price_second_excel > 0:  # Ensure there is no division by zero
                category_percentage_second[category] = (category_price_sums_second[category] / total_price_second_excel) * 100

        print('Last to Last Month Analysis')
        for category, percentage in category_percentage_second.items():
            print(f"Category optional'{category}' makes up {percentage:.2f}% of the Last to Last Month total price.")
        print()
        # Replace 0 with NaN to avoid division by zero error
        combined_df['Quantity Sold Second Excel'] = combined_df['Quantity Sold Second Excel'].replace(0, np.nan)

        # Now perform the division; any division by NaN will result in NaN instead of an error
        combined_df['Price ASP Second'] = combined_df['Price Second Excel'] / combined_df['Quantity Sold Second Excel']

        # If you want to handle NaN values in 'Price ASP Second', you can fill them with a default value
        # For example, filling NaN with 0 or any other placeholder value
        combined_df['Price ASP Second'] = combined_df['Price ASP Second'].fillna(0)

        category_price_sums_second1 = combined_df.groupby('User Category')['Price Second Excel'].sum()

        category_percentage_second1 = {}

        for category in category_price_sums_second1.index:
            if total_price_second_excel > 0:  # Ensure there is no division by zero
                category_percentage_second1[category] = (category_price_sums_second1[category] / total_price_second_excel) * 100

        print('Last to Last Month Analysis')
        for category, percentage in category_percentage_second1.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last to Last Month total price.")
        print()

        ABV1 = total_price_second_excel/num_of_bills
        print('Average Basket Value: ',ABV1)
        print()

        ABS1 = total_item_sold_second_excel/num_of_bills
        print("Average Basket Sales: ",ABS1)
        print()

        AIV1 = total_price_second_excel/total_item_sold_second_excel
        print("Average Item Value: ",AIV1)
        print()

        sqft1 = total_price_second_excel/store_szie
        print("Average Item Value: ",sqft1)
        print()

        GwothAPIPrevious = (ABS1*AIV1*num_of_bills)
        
        category_price_sums_first121 = combined_df.groupby('User Category')['Quantity Sold Second Excel'].sum()
        category_percentage_first121 = {}

        # Calculate the percentage for each category in first and second excel
        for category in category_price_sums_first121.index:
            category_percentage_first121[category] = (category_price_sums_first121[category] / total_item_sold) * 100

        
        table_data201 = [['User Category', 'Value%', 'Volume%']]
        for category in category_price_sums_second1:
            value_percent = category_price_sums_second1.get(category, 0)  # Get value percent, default to 0 if not found
            volume_percent = category_percentage_first121.get(category, 0)  # Get volume percent, default to 0 if not found
            table_data201.append([category, f"{value_percent:.2f}%", f"{volume_percent:.2f}%"])


        GrowthAPI = ((GwothAPICuurent - GwothAPIPrevious)/GwothAPIPrevious)*100

        #@@@@@@@@@@@@@@
        # Destination Food % for One Month
        # print("Destination Food for previous Month")
        destination_food_percentage = category_percentage_second.get('Destination Food', 0)
        destination_food_message = ""
        if destination_food_percentage > 45:
            destination_food_message = f"Destination categories which are core to your business have a strong contribution at {destination_food_percentage:.2f}%. "
        elif destination_food_percentage <= 45:
            destination_food_message = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers. You can do it by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions"
        # print("Destination food message for Previous month")
        # print(destination_food_message)
        # print()

        # if optional_filename:
        destination_food_items_previous_Month = combined_df[(combined_df['Category Type'] == 'Destination Food') & 
                                        (combined_df['Quantity Sold Second Excel'].notna()) &
                                        (combined_df['Quantity Sold Second Excel'] > 0) &
                                        (combined_df['Price Second Excel'].notna()) &
                                        (combined_df['Price Second Excel'] > 0)]

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"destination_food_items_previous_Month_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        destination_food_items_previous_Month.to_excel(filepath, index=False)


        # Calculate the combined percentage of Destination Food and Routine Non Core Food
        # print("All Food Categories for previous Month")
        combined_percentage = category_percentage_second.get('Destination Food', 0) + category_percentage_second.get('Routine Non Core Food', 0)
        substantially_below = "substantially " if combined_percentage < 50 else ""
        food_categories_message = ""

        if combined_percentage > 65:
            food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
        else:
            food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall {substantially_below}the desired level and is at {combined_percentage:.2f}%. Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        # print("food categories message message for Previous month")
        # print(food_categories_message)
        # print()
        # if optional_filename:
        food_categories_items_previous_Month = combined_df[
                (combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])) &
                (combined_df['Quantity Sold Second Excel'].notna()) &
                (combined_df['Quantity Sold Second Excel'] > 0) &
                (combined_df['Price Second Excel'].notna()) &
                (combined_df['Price Second Excel'] > 0)
            ]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"food_categories_items_previous_Month_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        food_categories_items_previous_Month.to_excel(filepath, index=False)
        
        routine_non_food_percentage = category_percentage_second.get('Routine Non Food', 0)
        routine_non_food_message = ""

        if routine_non_food_percentage < 30:
            routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        else:
            routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%."
        # print("Non Food Categories for Previous month")
        # print(routine_non_food_message)
        # print()
        # if optional_filename:
        routine_non_food_items_previous_Month = combined_df[(combined_df['Category Type'] == 'Routine Non Food') & 
                                        (combined_df['Quantity Sold Second Excel'].notna()) &
                                        (combined_df['Quantity Sold Second Excel'] > 0) &
                                        (combined_df['Price Second Excel'].notna()) &
                                        (combined_df['Price Second Excel'] > 0)]

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"routine_non_food_items_previous_Month{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        routine_non_food_items_previous_Month.to_excel(filepath, index=False)
        
    

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


    # Destination Food % for One Month
    # print("Destination Food for Last Month")
    destination_food_percentage = category_percentage_first.get('Destination Food', 0)
    destination_food_message = ""
    destination_insights = ""
    if destination_food_percentage > 45:
        destination_food_message = ""
    elif destination_food_percentage <= 45:
        destination_insights = f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%. "
        destination_food_message = (
    f"Destination categories which are core to your business have less than desired contribution at {destination_food_percentage:.2f}%."
    " Consider strengthening destination categories to bring them to at least 45% contribution. It can even go up to 50% depending upon shopping habits of your core customers."
    " You can do it by:\n"
    "Enhancing your product range\n"
    "Ensuring good display\n"
    "Ensuring product availability\n"
    "Planning promotions"
)
    print(destination_food_message)
    print()

    if mandatory_filename and destination_food_percentage <= 45:
        destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
        total_price_first_excel1 = combined_df['Price First Excel'].sum()
        total_item_sold = combined_df['Quantity Sold First Excel'].sum()
        category_price_sums_first10 = destination_food_df.groupby('Category Type')['Price First Excel'].sum()
        # category_percentage_first = {}
        # Generate a unique timestamp to append to the filename
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Destination_Food_Items_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        destination_food_df.to_excel(filepath, index=False)
        
        destination_food_df = combined_df[combined_df['Category Type'] == 'Destination Food']
        total_price_destination = combined_df['Price First Excel'].sum()
        category_price_sums_destination = destination_food_df.groupby('User Category')['Price First Excel'].sum()
        category_percentage_destinationNew = {category: (price / total_price_destination) * 100 for category, price in category_price_sums_destination.items()}
        for category, percentage in category_percentage_destinationNew.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
        data = {
            "User Category": list(category_percentage_destinationNew.keys()),
            "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destinationNew.values()]
        }
        df = pd.DataFrame(data)
    else:
        category_percentage_destinationNew = {}

    
    # Calculate the combined percentage of Destination Food and Routine Non Core Food
    print("All Food Categories for Last Month")
    combined_percentage = category_percentage_first.get('Destination Food', 0) + category_percentage_first.get('Routine Non Core Food', 0)
    substantially_below = "substantially " if combined_percentage < 50 else ""
    food_categories_message = ""
    food_categories_insights = ""

    if combined_percentage > 65:
        food_categories_message = ""
    else:
        food_categories_insights = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall the desired level and is at {combined_percentage:.2f}%. "
        food_categories_message = f"Shopping of food categories is the key motivation for monthly shopping of a family and they spend about 2/3rd on food categories. Your Food categories fall {substantially_below}the desired level and is at {combined_percentage:.2f}%. Consider strengthening of your food categories to improve sales and customer loyalty by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
    # print("food categories message message for Last month")
    print(food_categories_message)
    print()

    if mandatory_filename and combined_percentage < 65:
        food_categories_df = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
        # Generate a unique timestamp to append to the filename
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"food_categories_Items_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        food_categories_df.to_excel(filepath, index=False)

        food_categories_df100 = combined_df[combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])]
        total_price_destination100 = combined_df['Price First Excel'].sum()
        category_price_sums_destination100 = food_categories_df100.groupby('User Category')['Price First Excel'].sum()
        category_percentage_destination100 = {category: (price / total_price_destination100) * 100 for category, price in category_price_sums_destination100.items()}
        for category, percentage in category_percentage_destination100.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
        data = {
            "User Category": list(category_percentage_destination100.keys()),
            "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination100.values()]
        }
        df = pd.DataFrame(data)
    else:
        category_percentage_destination100 = {}


    print("Non Food Categories for Last Month")
    routine_non_food_percentage = category_percentage_first.get('Routine Non Food', 0)
    routine_non_food_message = ""
    routine_non_food_insights = ""

    if routine_non_food_percentage < 30:
        routine_non_food_insights = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
        routine_non_food_message = f"Your non-food categories contribution is at {routine_non_food_percentage:.2f}%. There is a possibility that you are losing the opportunity of selling these categories. Consider strengthening these categories without harming your food sales by -\nEnhancing your product range\nEnsuring good display\nEnsuring product availability\nPlanning promotions\nPremiumisation"
    else:
        routine_non_food_message = ""
    # print("routine non food message message message for Last month")
    print(routine_non_food_message)
    print()
    if mandatory_filename and routine_non_food_percentage < 30:
        Routine_Non_Food_df = combined_df[combined_df['Category Type'] == 'Routine Non Food']
        # Generate a unique timestamp to append to the filename
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Routine_Non_Food_Items_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        Routine_Non_Food_df.to_excel(filepath, index=False)

        Routine_Non_Food_df101 = combined_df[combined_df['Category Type'] == 'Routine Non Food']
        total_price_destination101 = combined_df['Price First Excel'].sum()
        category_price_sums_destination101 = Routine_Non_Food_df101.groupby('User Category')['Price First Excel'].sum()
        category_percentage_destination101 = {category: (price / total_price_destination101) * 100 for category, price in category_price_sums_destination101.items()}
        for category, percentage in category_percentage_destination101.items():
            print(f"User Category '{category}' makes up {percentage:.2f}% of the Last Month total price within Destination Food.")
        data = {
            "User Category": list(category_percentage_destination101.keys()),
            "Percentage": [f"{percentage:.2f}%" for percentage in category_percentage_destination101.values()]
        }
        df = pd.DataFrame(data)
    else:
        category_percentage_destination101 = {}

    if mandatory_filename and optional_filename:
        total_quantity_first_overall = combined_df['Quantity Sold First Excel'].sum()
        total_quantity_second_overall = combined_df['Quantity Sold Second Excel'].sum()
        total_price_first_overall = combined_df['Price First Excel'].sum()
        total_price_second_overall = combined_df['Price Second Excel'].sum()

        # Safeguard against division by zero
        if total_quantity_first_overall > 0 and total_price_first_overall > 0:
            quantity_percentage100_overall = ((total_quantity_first_overall-total_quantity_second_overall)/ total_quantity_second_overall ) * 100
            price_percentage100_overall = ((total_price_first_overall-total_price_second_overall) / total_price_second_overall) * 100
            print('Growing and degrowing Groups')
            print(f"Volume Growth % for Destination Food : {quantity_percentage100_overall:.2f}%")
            print(f"Price Growth % for Destination Food: {price_percentage100_overall:.2f}%")
        else:
            print("Insufficient data for calculating percentages.")

        # Determine the growth category based on quantity_percentage
        if quantity_percentage100_overall < 1:
            print("No Volume growth for Destination Food is seen.")
        elif quantity_percentage100_overall < 5:
            print(f"Volume growth for Destination Food is at {quantity_percentage100_overall:.2f}% which is positive but small.")
        elif quantity_percentage100_overall < 10:
            print(f"Volume growth for Destination Food is at {quantity_percentage100_overall:.2f}% which is excellent.")
        elif quantity_percentage100_overall > 20:
            print(f"Volume growth for Destination Food is at {quantity_percentage100_overall:.2f}% which is exceptional.")
        elif -5 <= quantity_percentage100_overall < 0:
            print(f"Volume degrowth for Destination Food is {quantity_percentage100_overall:.2f}%.")
        elif -10 <= quantity_percentage100_overall < -5:
            print(f"Volume degrowth for Destination Food is high at {quantity_percentage100_overall:.2f}%.")
        elif quantity_percentage100_overall < -10:
            print(f"Volume degrowth for Destination Food is very high at {quantity_percentage100_overall:.2f}%.")
        else:
            print(f"Growth rate for Destination Food is  {quantity_percentage100_overall:.2f}%.")
        
        # Determine the growth category based on quantity_percentage
        if price_percentage100_overall < 1:
            print("No value growth for Destination Food is seen.")
        elif price_percentage100_overall < 5:
            print(f"Value growth for Destination Food is at {price_percentage100_overall:.2f}% which is positive but small.")
        elif price_percentage100_overall < 10:
            print(f"Value growth for Destination Food is at {price_percentage100_overall:.2f}% which is excellent.")
        elif price_percentage100_overall > 20:
            print(f"Value growth for Destination Food is at {price_percentage100_overall:.2f}% which is exceptional.")
        elif -5 <= price_percentage100_overall < 0:
            print(f"Value degrowth for Destination Food is {price_percentage100_overall:.2f}%.")
        elif -10 <= price_percentage100_overall < -5:
            print(f"Value degrowth for Destination Food is high at {price_percentage100_overall:.2f}%.")
        elif price_percentage100_overall < -10:
            print(f"Value degrowth for Destination Food is very high at {price_percentage100_overall:.2f}%.")
        else:
            print(f"Growth rate for Destination Food is  {price_percentage100_overall:.2f}%.")
        print()




    if mandatory_filename and optional_filename:
        # Filter for 'Destination Food' where both quantities and prices are present
        destination_food_df = combined_df[(combined_df['Category Type'] == 'Destination Food') & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

        # Calculate sums and percentages
        total_quantity_first = destination_food_df['Quantity Sold First Excel'].sum()
        total_quantity_second = destination_food_df['Quantity Sold Second Excel'].sum()
        total_price_first = destination_food_df['Price First Excel'].sum()
        total_price_second = destination_food_df['Price Second Excel'].sum()

        # Safeguard against division by zero
        if total_quantity_first > 0 and total_price_first > 0:
            quantity_percentage100 = ((total_quantity_first-total_quantity_second)/ total_quantity_second ) * 100
            price_percentage100 = ((total_price_first-total_price_second) / total_price_second) * 100
            print('Growing and degrowing Groups')
            print(f"Volume Growth % for Destination Food : {quantity_percentage100:.2f}%")
            print(f"Price Growth % for Destination Food: {price_percentage100:.2f}%")
        else:
            print("Insufficient data for calculating percentages.")

        # Determine the growth category based on quantity_percentage
        if quantity_percentage100 < 1:
            print("No Volume growth for Destination Food is seen.")
        elif quantity_percentage100 < 5:
            print(f"Volume growth for Destination Food is at {quantity_percentage100:.2f}% which is positive but small.")
        elif quantity_percentage100 < 10:
            print(f"Volume growth for Destination Food is at {quantity_percentage100:.2f}% which is excellent.")
        elif quantity_percentage100 > 20:
            print(f"Volume growth for Destination Food is at {quantity_percentage100:.2f}% which is exceptional.")
        elif -5 <= quantity_percentage100 < 0:
            print(f"Volume degrowth for Destination Food is {quantity_percentage100:.2f}%.")
        elif -10 <= quantity_percentage100 < -5:
            print(f"Volume degrowth for Destination Food is high at {quantity_percentage100:.2f}%.")
        elif quantity_percentage100 < -10:
            print(f"Volume degrowth for Destination Food is very high at {quantity_percentage100:.2f}%.")
        else:
            print(f"Growth rate for Destination Food is  {quantity_percentage100:.2f}%.")
        
        # Determine the growth category based on quantity_percentage
        if price_percentage100 < 1:
            print("No value growth for Destination Food is seen.")
        elif price_percentage100 < 5:
            print(f"Value growth for Destination Food is at {price_percentage100:.2f}% which is positive but small.")
        elif price_percentage100 < 10:
            print(f"Value growth for Destination Food is at {price_percentage100:.2f}% which is excellent.")
        elif price_percentage100 > 20:
            print(f"Value growth for Destination Food is at {price_percentage100:.2f}% which is exceptional.")
        elif -5 <= price_percentage100 < 0:
            print(f"Value degrowth for Destination Food is {price_percentage100:.2f}%.")
        elif -10 <= price_percentage100 < -5:
            print(f"Value degrowth for Destination Food is high at {price_percentage100:.2f}%.")
        elif price_percentage100 < -10:
            print(f"Value degrowth for Destination Food is very high at {price_percentage100:.2f}%.")
        else:
            print(f"Growth rate for Destination Food is  {price_percentage100:.2f}%.")
        print()


    if mandatory_filename and optional_filename:
        # Filter for both 'Destination Food' and 'Routine Non Core Food' categories
        filtered_df = combined_df[(combined_df['Category Type'].isin(['Destination Food', 'Routine Non Core Food'])) & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

        # Calculate sums
        total_quantity_first = filtered_df['Quantity Sold First Excel'].sum()
        total_quantity_second = filtered_df['Quantity Sold Second Excel'].sum()
        total_price_first = filtered_df['Price First Excel'].sum()
        total_price_second = filtered_df['Price Second Excel'].sum()

        # Calculate and print percentage changes, ensuring no division by zero
        if total_quantity_first > 0 and total_price_first > 0:
            quantity_sold_percentage_change = ((total_quantity_first-total_quantity_second) / total_quantity_second) * 100
            price_percentage_change = ((total_price_first-total_price_second)/ total_price_second ) * 100

            print(f"Volume Growth % Change for Destination Food and Routine Non Core Food: {quantity_sold_percentage_change:.2f}%")
            print(f"Price Growth % Change for Destination Food and Routine Non Core Food: {price_percentage_change:.2f}%")
        else:
            print("Insufficient data for calculating percentage changes for Destination Food and Routine Non Core Food.")

        # Determine the growth category based on quantity_percentage
        if quantity_sold_percentage_change < 1:
            print("No Volume growth for Destination Food and Routine Non Core Food is seen.")
        elif quantity_sold_percentage_change < 5:
            print(f"Volume growth for Destination Food and Routine Non Core Food is at {quantity_sold_percentage_change:.2f}% which is positive but small.")
        elif quantity_sold_percentage_change < 10:
            print(f"Volume growth for Destination Food and Routine Non Core Food is at {quantity_sold_percentage_change:.2f}% which is excellent.")
        elif quantity_sold_percentage_change > 20:
            print(f"Volume growth for Destination Food and Routine Non Core Food is at {quantity_sold_percentage_change:.2f}% which is exceptional.")
        elif -5 <= quantity_sold_percentage_change < 0:
            print(f"Volume degrowth for Destination Food and Routine Non Core Food is {quantity_sold_percentage_change:.2f}%.")
        elif -10 <= quantity_sold_percentage_change < -5:
            print(f"Volume degrowth for Destination Food and Routine Non Core Food is high at {quantity_sold_percentage_change:.2f}%.")
        elif quantity_sold_percentage_change < -10:
            print(f"Volume degrowth for Destination Food and Routine Non Core Food is very high at {quantity_sold_percentage_change:.2f}%.")
        else:
            print(f"Growth rate for Destination Food and Routine Non Core Food is  {quantity_sold_percentage_change:.2f}%.")

        # Determine the growth category based on quantity_percentage
        if price_percentage_change < 1:
            print("No value growth for Destination Food and Routine Non Core Food is seen.")
        elif price_percentage_change < 5:
            print(f"Value growth for Destination Food and Routine Non Core Food is at {price_percentage_change:.2f}% which is positive but small.")
        elif price_percentage_change < 10:
            print(f"Value growth for Destination Food and Routine Non Core Food is at {price_percentage_change:.2f}% which is excellent.")
        elif price_percentage_change > 20:
            print(f"Value growth for Destination Food and Routine Non Core Food is at {price_percentage_change:.2f}% which is exceptional.")
        elif -5 <= price_percentage_change < 0:
            print(f"Value degrowth for Destination Food and Routine Non Core Food is {price_percentage_change:.2f}%.")
        elif -10 <= price_percentage_change < -5:
            print(f"Value degrowth for Destination Food and Routine Non Core Food is high at {price_percentage_change:.2f}%.")
        elif price_percentage_change < -10:
            print(f"Value degrowth for Destination Food and Routine Non Core Food is very high at {price_percentage_change:.2f}%.")
        else:
            print(f"Growth rate for Destination Food and Routine Non Core Food is  {price_percentage_change:.2f}%.")
        print()


    if mandatory_filename and optional_filename:
        # Filter for 'Routine Non Food' where both quantities and prices are present
        destination_food_df = combined_df[(combined_df['Category Type'] == 'Routine Non Food') & pd.notna(combined_df['Quantity Sold First Excel']) & pd.notna(combined_df['Quantity Sold Second Excel']) & pd.notna(combined_df['Price First Excel']) & pd.notna(combined_df['Price Second Excel'])]

        # Calculate sums and percentages
        total_quantity_first = destination_food_df['Quantity Sold First Excel'].sum()
        total_quantity_second = destination_food_df['Quantity Sold Second Excel'].sum()
        total_price_first = destination_food_df['Price First Excel'].sum()
        total_price_second = destination_food_df['Price Second Excel'].sum()

        # Safeguard against division by zero
        if total_quantity_first > 0 and total_price_first > 0:
            quantity_percentage102 = ((total_quantity_first-total_quantity_second) / total_quantity_second) * 100
            price_percentage102 = ((total_price_first-total_price_second) / total_price_second) * 100

            print(f"Volume Growth % for Routine Non Food : {quantity_percentage102:.2f}%")
            print(f"Price Growth % for Routine Non Food: {price_percentage102:.2f}%")
        else:
            print("Insufficient data for calculating percentages.")

        # Determine the growth category based on quantity_percentage
        if quantity_percentage102 < 1:
            print("No Volume growth for Routine Non Food is seen.")
        elif quantity_percentage102 < 5:
            print(f"Volume growth for Routine Non Food is at {quantity_percentage102:.2f}% which is positive but small.")
        elif quantity_percentage102 < 10:
            print(f"Volume growth for Routine Non Food is at {quantity_percentage102:.2f}% which is excellent.")
        elif quantity_percentage102 > 20:
            print(f"Volume growth for Routine Non Food is at {quantity_percentage102:.2f}% which is exceptional.")
        elif -5 <= quantity_percentage102 < 0:
            print(f"Volume degrowth for Routine Non Food is {quantity_percentage102:.2f}%.")
        elif -10 <= quantity_percentage102 < -5:
            print(f"Volume degrowth for Routine Non Food is high at {quantity_percentage102:.2f}%.")
        elif quantity_percentage102 < -10:
            print(f"Volume degrowth for Routine Non Food is very high at {quantity_percentage102:.2f}%.")
        else:
            print(f"Growth rate for Routine Non Food is  {quantity_percentage102:.2f}%.")
        
        # Determine the growth category based on quantity_percentage
        if price_percentage102 < 1:
            print("No Value growth for Routine Non Food is seen.")
        elif price_percentage102 < 5:
            print(f"Value growth for Routine Non Food is at {price_percentage102:.2f}% which is positive but small.")
        elif price_percentage102 < 10:
            print(f"Value growth for Routine Non Food is at {price_percentage102:.2f}% which is excellent.")
        elif price_percentage102 > 20:
            print(f"Value growth for Routine Non Food is at {price_percentage102:.2f}% which is exceptional.")
        elif -5 <= price_percentage102 < 0:
            print(f"Value degrowth for Routine Non Food is {price_percentage102:.2f}%.")
        elif -10 <= price_percentage102 < -5:
            print(f"Value degrowth for Routine Non Food is high at {price_percentage102:.2f}%.")
        elif price_percentage102 < -10:
            print(f"Value degrowth for Routine Non Food is very high at {price_percentage102:.2f}%.")
        else:
            print(f"Growth rate for Routine Non Food is  {price_percentage102:.2f}%.")
        print()


    # Salt Circle Analysis
    # Step 1: Filter combined_df for Sugar/Salt category and Salt product
    salt_circle_df = combined_df[(combined_df['Category Predicted'] == 'Sugar/Salt') & (combined_df['Product Predicted'] == 'Salt')]

    # Step 2: Calculate No_of_Salt_Packets
    No_of_Salt_Packets = salt_circle_df['Quantity Sold First Excel'].sum()

    # Check to avoid division by zero
    if No_of_Salt_Packets > 0:
        # Step 3: Calculate No of Families Shopping Estimate
        Total_Sales = combined_df['Price First Excel'].sum()
        No_of_Families_Shopping_Estimate = Total_Sales / No_of_Salt_Packets
    else:
        No_of_Families_Shopping_Estimate = 0

    # Fixed value for Average Monthly Basket Estimate
    Average_Monthly_Basket_Estimate = 4000  # Rs

    # Step 4: Compare and determine if it's greater or less
    if No_of_Families_Shopping_Estimate > Average_Monthly_Basket_Estimate:
        salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
        comparison_message = f"The No of Families Shopping Estimate is greater than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate:.2f}"
    elif No_of_Families_Shopping_Estimate < Average_Monthly_Basket_Estimate:
        salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."
        comparison_message = f"The No of Families Shopping Estimate is less than the Average Monthly Basket Estimate which is Rs{No_of_Families_Shopping_Estimate}"
    else:
        salt_msg = f"No. of family shopping is {No_of_Salt_Packets}. Average Monthly Basket Estimate is Rs{No_of_Families_Shopping_Estimate:.2f}. Total Sales is Rs{Total_Sales}."

    print(comparison_message)


    if mandatory_filename:
        total_sku = len(combined_df)
        total_sales = combined_df['Price First Excel'].sum()
        df_sorted = combined_df.sort_values(by='Price First Excel', ascending=False)
        df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
        df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
        cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
        top_80_percent_items = df_sorted.loc[:cutoff_index]
        num_top_80_percent_items = len(top_80_percent_items)
        percentageofsku = (num_top_80_percent_items / total_sku) * 100
        SKU_msg = f"Total SKU is {total_sku}. Total Sales is Rs{total_sales:.2f}. {num_top_80_percent_items} SKU's contribute to 80% of sales which is {percentageofsku:.2f}% of total SKU's."
        # output_path = 'Top_80_Percent_High_Selling_Sku.xlsx'
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'SKU_Contributing_80%_sales_{timestamp}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        top_80_percent_items.to_excel(output_path, index=False)

        top_20_items = top_80_percent_items.head(20)[['Item Name', 'Price First Excel']]

        top_20_items = top_20_items.rename(columns={
            'Price First Excel': 'Price Last Month',
        })
        data103 = [top_20_items.columns.tolist()] + top_20_items.values.tolist()


    if mandatory_filename:
        top20product = combined_df.groupby('Product Predicted').agg({
            'Quantity Sold First Excel': 'sum',
            'Price First Excel': 'sum',
        }).reset_index()
        top20product_top = top20product.sort_values(by='Price First Excel', ascending=False).head(20)
        top20product_bottom = top20product.sort_values(by='Price First Excel', ascending=True).head(20)

    

    if mandatory_filename:
        aggregated_data = combined_df.groupby('Product Predicted').agg({
            'Quantity Sold First Excel': 'sum',
            'Price First Excel': 'sum',
        }).reset_index()
        total_sales = aggregated_data['Price First Excel'].sum()
        df_sorted = aggregated_data.sort_values(by='Price First Excel', ascending=False)
        df_sorted['cumulative_sales'] = df_sorted['Price First Excel'].cumsum()
        df_sorted['cumulative_percentage'] = df_sorted['cumulative_sales'] / total_sales * 100
        cutoff_index = df_sorted[df_sorted['cumulative_percentage'] >= 80].index[0]
        top_80_percent_items = df_sorted.loc[:cutoff_index]
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'Product_Contributing_80%_sales_{timestamp}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        top_80_percent_items.to_excel(output_path, index=False)

    if mandatory_filename and optional_filename:
        aggregated_data1 = combined_df.groupby('Product Predicted').agg({
            'Quantity Sold First Excel': 'sum',
            'Quantity Sold Second Excel': 'sum',   
        }).reset_index()
        aggregated_data1['Volume Growth'] = (aggregated_data1['Quantity Sold First Excel'] / aggregated_data1['Quantity Sold Second Excel']) * 100
        aggregated_data_Top_50 = aggregated_data1.sort_values(by='Volume Growth', ascending=False).head(50)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'Top_50_Product_volume_growth_{timestamp}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        aggregated_data_Top_50.to_excel(output_path, index=False)
        aggregated_data_Bottom_50 = aggregated_data1.sort_values(by='Volume Growth', ascending=True).head(50)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'Top_50_Product_volume_De-Growth_{timestamp}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
        aggregated_data_Bottom_50.to_excel(output_path, index=False)
        # Prepare data for PDF (Top 20 only from each)
        top_20_growth_data_Product = aggregated_data_Top_50.head(20)
        bottom_20_de_growth_data_Product = aggregated_data_Bottom_50.head(20)

        top_20_growth_data_Product_display = top_20_growth_data_Product.rename(columns={
            'Product Predicted': "Product",
            'Quantity Sold First Excel': 'Quantity Sold in Last Month',
            'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
        })

        bottom_20_de_growth_data_Product_display = bottom_20_de_growth_data_Product.rename(columns={
            'Quantity Sold First Excel': 'Quantity Sold in Last Month',
            'Quantity Sold Second Excel': 'Quantity Sold in Previous Month'
        })


    if mandatory_filename and optional_filename:
        # Assuming combined_df is already defined and filled with the necessary data
        # Step 1: Aggregate the data
        aggregated_data = combined_df.groupby('Product Predicted').agg({
            'Quantity Sold First Excel': 'sum',
            'Price First Excel': 'sum',
            'Quantity Sold Second Excel': 'sum',
            'Price Second Excel': 'sum'
        }).reset_index()

        # Step 2: Calculate total sales
        aggregated_data['Total Sales First Excel'] = aggregated_data['Price First Excel']
        aggregated_data['Total Sales Second Excel'] = aggregated_data['Price Second Excel']

        # Calculate the total sales from both excels
        total_sales_first = aggregated_data['Total Sales First Excel'].sum()
        total_sales_second = aggregated_data['Total Sales Second Excel'].sum()

        # Step 3: Sort data by total sales in descending order and calculate cumulative percentage
        aggregated_data_sorted_first = aggregated_data.sort_values(by='Total Sales First Excel', ascending=False)
        aggregated_data_sorted_second = aggregated_data.sort_values(by='Total Sales Second Excel', ascending=False)

        # Calculate cumulative sums
        aggregated_data_sorted_first['Cumulative Sales First'] = aggregated_data_sorted_first['Total Sales First Excel'].cumsum()
        aggregated_data_sorted_second['Cumulative Sales Second'] = aggregated_data_sorted_second['Total Sales Second Excel'].cumsum()

        # Calculate cumulative percentage
        aggregated_data_sorted_first['Cumulative Percentage First'] = 100 * aggregated_data_sorted_first['Cumulative Sales First'] / total_sales_first
        aggregated_data_sorted_second['Cumulative Percentage Second'] = 100 * aggregated_data_sorted_second['Cumulative Sales Second'] / total_sales_second

        # Find the number of products contributing to 80% of sales
        number_of_products_first = (aggregated_data_sorted_first['Cumulative Percentage First'] <= 80).sum()
        number_of_products_second = (aggregated_data_sorted_second['Cumulative Percentage Second'] <= 80).sum()

        # Filter the DataFrame for products contributing to top 80% of sales
        top_80_first = aggregated_data_sorted_first[aggregated_data_sorted_first['Cumulative Percentage First'] <= 80]
        top_80_second = aggregated_data_sorted_second[aggregated_data_sorted_second['Cumulative Percentage Second'] <= 80]

        # Combine these DataFrames to have a complete view of top contributing products
        # Optionally, you can merge them on 'Product Predicted' if they share similar products and you want to compare
        # top_80_combined = pd.merge(top_80_first, top_80_second, on='Product Predicted', suffixes=('_First', '_Second'), how='outer')

        # Save to Excel
        with pd.ExcelWriter('Top_80_Percent_Contributors.xlsx') as writer:
            top_80_first.to_excel(writer, sheet_name='Top 80% First Excel', index=False)
            top_80_second.to_excel(writer, sheet_name='Top 80% Second Excel', index=False)
            # top_80_combined.to_excel(writer, sheet_name='Combined Top 80%', index=False)

        # Print results
        print("Number of products contributing to 80% of sales in First Excel:", number_of_products_first)
        print("Number of products contributing to 80% of sales in Second Excel:", number_of_products_second)



    #SKU Sales Performance Top 50 SKU volume growth & SKU Sales Performance Bottom 50 SKU volume degrowth
    if mandatory_filename and optional_filename:
        combined_df['Quantity Growth'] = pd.to_numeric(combined_df['Quantity Growth'].str.rstrip('%'), errors='coerce')
        sorted_combined_df = combined_df.sort_values(by='Quantity Growth', ascending=False).head(50)
        top_50_SKU = f"Top50SKU_Volume_Growth_{uuid4()}.xlsx"
        top_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], top_50_SKU)   
        sorted_combined_df.to_excel(top_50_file_path, index=False)

        sorted_combined_df_degrowth = combined_df.sort_values(by='Quantity Growth', ascending=True).head(50)
        bottom_50_products_degrowth = sorted_combined_df_degrowth.drop_duplicates(subset=['Product Predicted'], keep='first').head(50)
        bottom_50_SKU = f"Bottom50SKU_Volume_De-Growth_{uuid4()}.xlsx"
        bottom_50_file_path = os.path.join(app.config['UPLOAD_FOLDER'], bottom_50_SKU)
        bottom_50_products_degrowth.to_excel(bottom_50_file_path, index=False)

        # Prepare top 20 growth and de-growth data for PDF
        top_20_growth_data = sorted_combined_df.head(20)
        columns_to_keep = ['Item Name', 'Quantity Sold First Excel', 'Quantity Sold Second Excel','Quantity Growth']
        top_20_growth_data = top_20_growth_data.loc[:, columns_to_keep]
        bottom_20_degrowth_data = bottom_50_products_degrowth.head(20)
        bottom_20_degrowth_data = bottom_20_degrowth_data.loc[:, columns_to_keep]

        # SKUs with less than 5 units sale IN BOTH MONTHS
        less_then_5_sku =  combined_df[(combined_df['Quantity Sold First Excel'] < 5) & (combined_df['Quantity Sold Second Excel'] < 5)]
        unique_filename = f'less_then_5_sku_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        less_then_5_sku.to_excel(output_path, index=False)

        # # Filter and collect details for SKUs sold less than 5 times in the mandatory file
        # mandatory_filter = combined_df[combined_df['Quantity Sold First Excel'] < 5]
        # # Similar for optional, but make sure to adjust column names as needed
        # optional_filter = combined_df[combined_df['Quantity Sold Second Excel'] < 5]
        # # Initialize combined_df with specified columns
        # combined_df5 = pd.DataFrame(columns=[
        #     'Item Name', 'User Category', 'Category Predicted', 'Product Predicted',
        #     'Brand Predicted', 'Category Type', 'Quantity Sold First Excel',
        #     'Price First Excel', 'Quantity Sold Second Excel',
        #     'Price Second Excel', 'Quantity Growth', 'Value Growth'
        # ])

        # # Process mandatory_filter DataFrame
        # new_rows = []
        # for _, row in mandatory_filter.iterrows():
        #     new_row =  {
        #         'Item Name': row['Item Name'],
        #         # Add other details similarly
        #         'Quantity Sold First Excel': row['Quantity Sold First Excel'],
        #         'Price First Excel': row['Price First Excel'],
        #         # Set optional fields to None or appropriate default
        #         'Quantity Sold Second Excel': None,
        #         'Price Second Excel': None
        #     }
        #     new_rows.append(new_row)
        # new_rows_df = pd.DataFrame(new_rows)
        # combined_df5 = pd.concat([combined_df5, new_rows_df], ignore_index=True)    
        # # Process optional_filter DataFrame
        # new_rows = []
        # for _, row in optional_filter.iterrows():
        #     if row['Item Name'] in combined_df5['Item Name'].values:
        #         # Update the existing row if SKU exists
        #         idx = combined_df5.index[combined_df5['Item Name'] == row['Item Name']].tolist()[0]
        #         combined_df5.at[idx, 'Quantity Sold Second Excel'] = row['Quantity Sold Second Excel']
        #         combined_df5.at[idx, 'Price Second Excel'] = row['Price Second Excel']
        #     else:
        #         # Append a new row if SKU doesn't exist
        #         new_row =  {
        #             'Item Name': row['Item Name'],
        #             'Quantity Sold First Excel': None,
        #             'Price First Excel': None,
        #             'Quantity Sold Second Excel': row['Quantity Sold Second Excel'],
        #             'Price Second Excel': row['Price Second Excel']
        #         }
        #         new_rows.append(new_row)
        # new_rows_df = pd.DataFrame(new_rows)
        # combined_df5 = pd.concat([combined_df5, new_rows_df], ignore_index=True)
        # output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'SKUs_Sold_Less_Than_5_Combined.xlsx')
        # combined_df5.to_excel(output_filename, index=False)



        # Step 1: Convert 'Quantity Growth' from percentage string to numeric format
        # Note: Assuming 'Quantity Growth' is calculated and formatted as a string with a percent sign
        combined_df['Quantity Growth'] = combined_df['Quantity Growth'].astype(str)
        combined_df['Quantity Growth Numeric'] = pd.to_numeric(combined_df['Quantity Growth'].str.rstrip('%'), errors='coerce') / 100

        # Step 2: Sort the DataFrame by 'Quantity Growth Numeric' in descending order
        sorted_combined_df = combined_df.sort_values(by='Quantity Growth Numeric', ascending=False)
        # Step 3: Select the top 50 SKUs based on Quantity Growth
        top_50_skus_by_growth = sorted_combined_df.head(50)
        # Optional: If you want to drop the 'Quantity Growth Numeric' column and keep the original 'Quantity Growth' as string
        top_50_skus_by_growth = top_50_skus_by_growth.drop(columns=['Quantity Growth Numeric'])
        # Save the top 50 SKUs to a new Excel file
        top_50_growth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Top_50_SKUs_by_Quantity_Growth.xlsx')
        top_50_skus_by_growth.to_excel(top_50_growth_filename, index=False)

        # Step 2: Sort the DataFrame by 'Quantity Growth Numeric' in ascending order to find the largest de-growth
        sorted_combined_df_for_degrowth = combined_df.sort_values(by='Quantity Growth Numeric', ascending=True)
        # Step 3: Select the top 50 SKUs based on Quantity De-growth
        top_50_skus_by_degrowth = sorted_combined_df_for_degrowth.head(50)
        # Save the top 50 SKUs with highest de-growth to a new Excel file
        top_50_degrowth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Top_50_SKUs_by_Quantity_DeGrowth.xlsx')
        top_50_skus_by_degrowth.to_excel(top_50_degrowth_filename, index=False)

    if mandatory_filename and optional_filename:
        save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')
        save_pie_chart(category_percentage_second, 'previous_month_pie_chart.png', 'Previous Month Category Percentages')
    else:
        save_pie_chart(category_percentage_first, 'last_month_pie_chart.png', 'Last Month Category Percentages')


    # Pdf Generation
    pdf_filename = f'Sales_Analysis_Report_{timestamp}.pdf'
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    elements = []
    elements1 = []

    if mandatory_filename and optional_filename:
        # Section : Purpose
        styles.add(ParagraphStyle(name='CenteredBoldHeading2', parent=styles['Heading2'], alignment=TA_CENTER, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='CenteredHeading1', parent=styles['Heading1'], alignment=TA_CENTER))

        # Header Section
        story.append(Spacer(1, 12))
        story.append(Paragraph("Sales Analysis Insight & Action Report", styles['CenteredBoldHeading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("A sales analysis tool is a software application that helps businesses track, analyze, and visualize their sales data to gain insights into sales performance. It typically offers features like sales forecasting, trend analysis, and performance metrics. These tools enable companies to make data-driven decisions, identify sales opportunities, optimize sales strategies, and enhance customer satisfaction. They are commonly used by sales teams, managers, and executives to monitor sales activities and achieve business goals.", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())

        # Section 1: Key Performance Indicators
        story.append(Paragraph("KEY Performance Indicators", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Basket Value (ABV): Average Value of each bill", styles['BodyText']))
        story.append(Paragraph(f"(a) Average Basket Value (ABV) ={ABV:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Basket Sales (ABS): Average number of items in each bill", styles['BodyText']))
        story.append(Paragraph(f"(b) Average Basket Sales (ABS) = {ABS:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Average Item Value (ASP): Average price of each item sold", styles['BodyText']))
        story.append(Paragraph(f"(c) Average Item Value (ASP) = {AIV:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(d) Per Sq. ft.sales = {sqft:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(e) Number of bills: {num_of_bills}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(f) Current Month sale Value: {total_price_first_excel:.2f}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"(g) Store Size: {store_szie}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Spacer(1, 12))
        story.append(Paragraph(" Family Basket", styles['Heading1']))
        story.append(Paragraph("Salt Circle", styles['BodyText']))
        story.append(Paragraph("This part of the report shows you the estimates of numbers of families who have shopped with you in the given month and their shopping basket assessment.", styles['BodyText']))
        story.append(Paragraph(f"{salt_msg}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())


        # Section 2: Business Group Performance
        story.append(Paragraph("Business Group Performance", styles['Heading1']))
        story.append(Paragraph("This part of the analysis shows you the performance of your various business groups that consist of similar categories in the way customers see them. It considers the benchmark participation of these business groups and provides you insights and actions required,", styles['BodyText']))
        last_month_image = Image('last_month_pie_chart.png')
        last_month_image._restrictSize(400, 400)
        story.append(last_month_image)
        if destination_food_message.strip():
            story.append(Paragraph("Destination Category %", styles['BodyText']))
            destination_insights1 = Paragraph(destination_insights, styles['BodyText'])
            story.append(destination_insights1)
            story.append(Spacer(1, 12))
            destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
            story.append(destination_food_paragraph)
            columns_to_display = [
                "Item Name",
                "Quantity Sold First Excel",
                "Price First Excel",
            ]
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destinationNew.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if food_categories_message.strip():
            story.append(Paragraph("All Food Category %", styles['BodyText']))
            line12 = Paragraph(food_categories_insights, styles['BodyText'])
            story.append(line12)
            story.append(Spacer(1, 12))
            line = Paragraph(food_categories_message, styles['BodyText'])
            story.append(line)
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination100.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if routine_non_food_message.strip():
            story.append(Paragraph("Non Food Category %", styles['BodyText']))
            line13 = Paragraph(routine_non_food_insights, styles['BodyText'])
            story.append(line13)
            story.append(Spacer(1, 12))
            line1 = Paragraph(routine_non_food_message, styles['BodyText'])
            story.append(line1)
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination101.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())



        # Section 1: Business Group Performance
        story.append(Paragraph("1. Business Group Performance", styles['Heading1']))
        story.append(Paragraph("This part of the analysis shows you the performance of your various business groups that consist of similar categories in the way customers see them. It considers the benchmark participation of these business groups and provides you insights and actions required,", styles['BodyText']))
        last_month_image = Image('last_month_pie_chart.png')
        last_month_image._restrictSize(400, 400)  # adjust size as needed
        story.append(last_month_image)
        
        if destination_food_message.strip():  # This checks if the message is not empty or just whitespace
            story.append(Paragraph("(a) Destination Category %", styles['BodyText']))
            destination_insights1 = Paragraph(destination_insights, styles['BodyText'])
            story.append(destination_insights1)
            story.append(Spacer(1, 12))
            destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
            story.append(destination_food_paragraph)  # Add to story
            story.append(Spacer(1, 12))
            columns_to_display = [
                "Item Name",
                "Quantity Sold First Excel",
                "Price First Excel",
            ]
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destinationNew.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if food_categories_message.strip():
            story.append(Paragraph("(b) All Food Category %", styles['BodyText']))
            line12 = Paragraph(food_categories_insights, styles['BodyText'])
            story.append(line12)
            story.append(Spacer(1, 12))
            line = Paragraph(food_categories_message, styles['BodyText'])
            story.append(line)
            story.append(Spacer(1, 12))
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination100.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
        if routine_non_food_message.strip():
            story.append(Paragraph("(c) Non Food Category %", styles['BodyText']))
            line13 = Paragraph(routine_non_food_insights, styles['BodyText'])
            story.append(line13)
            story.append(Spacer(1, 12))
            line1 = Paragraph(routine_non_food_message, styles['BodyText'])
            story.append(line1)
            story.append(Spacer(1, 12))
            table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination101.items()]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
        story.append(PageBreak())


        # Section 2: Business Group Growth
        story.append(Paragraph("Business Group Growth", styles['Heading1']))
        story.append(Paragraph("This part of the report analyses sales growth of different business groups. This helps you understand if the business groups participation is moving in the right direction against the benchmarks.", styles['BodyText']))
        story.append(Paragraph("(a) Overall Growth - Value Growth, Volume Growth", styles['Heading3']))
        story.append(Paragraph(f"Volume Growth % for Destination Food : {quantity_percentage100_overall:.2f}%", styles['BodyText']))
        story.append(Paragraph(f"Value Growth % for Destination Food: {price_percentage100_overall:.2f}%", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("(b) Destination Category - Value Growth, Volume Growth", styles['Heading3']))
        story.append(Paragraph(f"Volume Growth % for Destination Food : {quantity_percentage100:.2f}%", styles['BodyText']))
        story.append(Paragraph(f"Value Growth % for Destination Food: {price_percentage100:.2f}%", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("(c) Food Category - Value Growth, Volume Growth", styles['Heading3']))
        story.append(Paragraph(f"Volume Growth % for Food Category : {quantity_sold_percentage_change:.2f}%", styles['BodyText']))
        story.append(Paragraph(f"Value Growth % for Food Category: {price_percentage_change:.2f}%", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("(d) Non Food Growth - Value Growth, Volume Growth", styles['Heading3']))
        story.append(Paragraph(f"Volume Growth % for Non Food Growth : {quantity_percentage102:.2f}%", styles['BodyText']))
        story.append(Paragraph(f"Value Growth % for Non Food Growth: {price_percentage102:.2f}%", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 3: Category Growth
        story.append(Paragraph("3. Category Growth", styles['Heading2']))
        story.append(Paragraph("This part of the report shows you how your categories have performed as compared to the previous month.", styles['BodyText']))
        story.append(Paragraph(f"Last Month", styles['BodyText']))

        # elements = []

        # # Prepare the data for the table
        # table_data201 = [['User Category', 'Value%', 'Volume%']]
        # # Assuming you have filled `category_price_sums_second1` and `category_percentage_first121` dictionaries
        # for category in category_price_sums_second1:
        #     value_percent = category_price_sums_second1.get(category, 0)  # Get value percent, default to 0 if not found
        #     volume_percent = category_percentage_first121.get(category, 0)  # Get volume percent, default to 0 if not found
        #     table_data201.append([category, f"{value_percent:.2f}%", f"{volume_percent:.2f}%"])

        # # Create the table
        # table = Table(table_data201)
        # table.setStyle(TableStyle([
        #     ('BACKGROUND', (0,0), (-1,0), colors.gray),
        #     ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        #     ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        #     ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        #     ('BOTTOMPADDING', (0,0), (-1,0), 12),
        #     ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        #     ('GRID', (0,0), (-1,-1), 1, colors.black),
        # ]))
        # story.append(table)

        # Append the table to the list of elements that will be added to the document
        # elements.append(table)





        t = Table(table_data200)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.gray),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ]))
        story.append(t)
        # elements.append(t)
        story.append(Paragraph("(b) Previous month all category %", styles['BodyText']))
        t1 = Table(table_data201)
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.gray),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ]))
        story.append(t1)
        # elements1.append(t1)
        # # Table 1 Data     'User Category', 'Value%', 'Volume%'
        # table_data1 = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in table_data201.items()]
        # table1 = Table(table_data1)
        # table1.setStyle(TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        # ]))

        # # Table 2 Data
        # table_data2 = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_second1.items()]
        # table2 = Table(table_data2)
        # table2.setStyle(TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        # ]))

        # # Combine Tables Side by Side
        # combined_table_data = [[table1, table2]]
        # combined_table = Table(combined_table_data)

        # # Adjust table widths to fit page
        # combined_table._argW[0] = 200  # Width for table1
        # combined_table._argW[1] = 200  # Width for table2

        # # Add combined table to the story
        # story.append(combined_table)
        # table = Table(table_data)
        # table.setStyle(TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        # ]))
        # story.append(table)
        # story.append(Paragraph("(b) Previous month all category %", styles['BodyText']))
        # previous_month_image = Image('previous_month_pie_chart.png')
        # previous_month_image._restrictSize(300, 300)  # adjust size as needed
        # story.append(previous_month_image)
        # table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_second1.items()]
        # table = Table(table_data)
        # table.setStyle(TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        # ]))
        # story.append(table)
        # story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 4: Product Sales Performance
        story.append(Paragraph("4. Product Sales Performance", styles['Heading2']))
        story.append(Paragraph("This part of the report shows you business growth at product level", styles['BodyText']))
        add_table_to_story(top20product_top, "Top 20 sales contributing Products", story, styles)
        add_table_to_story(top20product_bottom, "Bottom 20 sales contributing products", story, styles)
        add_table_to_story(less_then_5_sku, "SKUs with less than 10 unit sales in the month", story, styles)
        add_table_to_story(top_20_growth_data_Product_display, "(a) Top 20 product Volume Growth", story, styles)
        add_table_to_story(bottom_20_de_growth_data_Product_display, "(b) Bottom 20 product Volume Growth", story, styles)
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 5: SKU Sales Growth
        story.append(Paragraph("5. SKU Sales Growth", styles['Heading2']))
        story.append(Paragraph("This part of the report shows you business growth at SKU level", styles['BodyText']))
        story.append(Paragraph("(a) Top 20 SKU Volume Growth", styles['BodyText']))
        add_table_to_story(top_20_growth_data, "(a) Top 20 SKU Volume Growth", story, styles)
        story.append(Paragraph("(b) Bottom 20 SKU Volume Growth", styles['BodyText']))
        add_table_to_story(bottom_20_degrowth_data, "(b) Bottom 20 SKU Volume Growth", story, styles)
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 6: Family Basket
        story.append(Paragraph("6. Family Basket", styles['Heading1']))
        story.append(Paragraph("(a) Salt Circle", styles['BodyText']))
        story.append(Paragraph("This part of the report shows you the estimates of numbers of families who have shopped with you in the given month and their shopping basket assessment.", styles['BodyText']))
        story.append(Paragraph(f"{salt_msg}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("(b) SKU contributing to 80% sales", styles['BodyText']))
        story.append(Paragraph("This part of the report shows you products and SKUs that constitute large part of your business", styles['BodyText']))
        story.append(Paragraph(f"{SKU_msg}", styles['BodyText']))
        table = Table(data103)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 8: Key Performance Indicators
        story.append(Paragraph("8. KEY Performance Indicators", styles['Heading2']))
        story.append(Paragraph("Average Basket Value (ABV): Average Value of each bill", styles['BodyText']))
        story.append(Paragraph(f"(a) Average Basket Value (ABV) ={ABV}", styles['BodyText']))
        story.append(Paragraph("Average Basket Sales (ABS): Average number of items in each bill", styles['BodyText']))
        story.append(Paragraph(f"(b) Average Basket Sales (ABS) = {ABS}", styles['BodyText']))
        story.append(Paragraph("Average Item Value (ASP): Average price of each item sold", styles['BodyText']))
        story.append(Paragraph(f"(c) Average Item Value (ASP) = {AIV}", styles['BodyText']))
        story.append(Paragraph(f"(d) Per Sq. ft.sales = {sqft}", styles['BodyText']))
        story.append(Paragraph(f"(e) Number of bills: {num_of_bills}", styles['BodyText']))
        story.append(Paragraph(f"(f) Current Month sale Value: {total_price_first_excel}", styles['BodyText']))
        story.append(Paragraph(f"(g) Store Size: {store_szie}", styles['BodyText']))

        # Build the PDF
        pdf.build(story)
    else:
        # Section 1: Purpose
        story.append(Paragraph("1. Purpose", styles['Heading1']))
        story.append(Paragraph("A sales analysis tool is a software application that helps businesses track, analyze, and visualize their sales data to gain insights into sales performance, It typically offers features like sales forecasting, trend analysis, and performance metrics. These tools enable companies to make data-driven decisions, identify sales opportunities, optimize sales strategies, and enhance customer satisfaction. They are commonly used by sales teams, managers, and executives to monitor sales activities and achieve business goals", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 2: Business Group Performance
        story.append(Paragraph("2. Business Group Performance", styles['Heading1']))
        last_month_image = Image('last_month_pie_chart.png')
        last_month_image._restrictSize(400, 400)  # adjust size as needed
        story.append(last_month_image)
        story.append(Paragraph("(a) Destination Category %", styles['BodyText']))
        destination_food_paragraph = Paragraph(destination_food_message, styles['BodyText'])
        story.append(destination_food_paragraph)  # Add to story
        columns_to_display = [
            "Item Name",
            "Quantity Sold First Excel",
            "Price First Excel",
        ]
        table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destinationNew.items()]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Paragraph("(b) All Food Category %", styles['BodyText']))
        line = Paragraph(food_categories_message, styles['BodyText'])
        story.append(line)
        table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination100.items()]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Paragraph("(c) Non Food Category %", styles['BodyText']))
        line1 = Paragraph(routine_non_food_message, styles['BodyText'])
        story.append(line1)
        table_data = [['User Category', 'Percentage']] + [(cat, f"{perc:.2f}%") for cat, perc in category_percentage_destination101.items()]
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 3: Family Basket
        story.append(Paragraph("3. Family Basket", styles['Heading1']))
        story.append(Paragraph("(a) Salt Circle", styles['BodyText']))
        story.append(Paragraph(f"{comparison_message}", styles['BodyText']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("(b) SKU contributing to 80% sales", styles['BodyText']))
        table = Table(data103)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(PageBreak())
        # Section 4: Key Performance Indicators
        story.append(Paragraph("4. KEY Performance Indicators", styles['Heading2']))
        story.append(Paragraph(f"(a) Average Basket Value (ABV) ={ABV}", styles['BodyText']))
        story.append(Paragraph(f"(b) Average Basket Sales (ABS) = {ABS}", styles['BodyText']))
        story.append(Paragraph(f"(c) Average Item Value (ASP) = {AIV}", styles['BodyText']))
        story.append(Paragraph(f"(d) Per Sq. ft.sales = {sqft}", styles['BodyText']))
        story.append(Paragraph(f"(e) Number of bills: {num_of_bills}", styles['BodyText']))
        story.append(Paragraph(f"(f) Current Month sale Value: {total_price_first_excel}", styles['BodyText']))
        story.append(Paragraph(f"(g) Store Size: {store_szie}", styles['BodyText']))

        # Build the PDF
        pdf.build(story)



    print(f"PDF generated: {pdf_filename}")

    flash('Processing complete. Structured data saved.')
    session.clear()
    return True 
#send_file(output_file_path, as_attachment=True, download_name=output_filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True,use_reloader=False)
