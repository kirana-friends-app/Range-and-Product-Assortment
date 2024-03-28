function submitForm() {
    const storeSize = document.getElementById("storeSize").value;

    fetch("http://13.51.121.171:5000/calculate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ storeSize: storeSize }),
    })
    .then(response => response.json())
    .then(data => {
        if (!data.totalSKUs) {
            throw new Error(`Server error: ${data.error}`);
        }
        console.log('Success:', data);
        document.querySelector(".output-container p").innerHTML = `Total SKU Count: ${data.totalSKUs}<br>Total Product Count: ${data.totalProductCount}<br>Unique Category Count: ${data.uniqueCategoryCount}`;

        // Call function to display products in a table
        displayProductsTable(data.products);
    })
    .catch(error => {
        console.error("Error:", error);
        document.querySelector(".output-container p").textContent = `Error: ${error}`;
    });
}

// New function to display products in a table
function displayProductsTable(products) {
    let html = '<table><tr><th>CATEGORY</th><th>PRODUCT</th><th>BRAND</th><th>DESCRIPTION</th></tr>';
    products.forEach(product => {
        html += `<tr><td>${product.CATEGORY}</td><td>${product.PRODUCT}</td><td>${product.BRAND}</td><td>${product['Product Description']}</td></tr>`;
    });
    html += '</table>';
    document.getElementById("excel-container").innerHTML = html;
}
