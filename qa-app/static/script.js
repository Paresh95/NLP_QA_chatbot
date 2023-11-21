// Function to handle form submission
function handleSubmit(formId, event) {
    event.preventDefault(); // Prevent default form submission

    const form = document.getElementById(formId);
    const formData = new FormData(form);

    let url = form.action;
    let fetchOptions = {
        method: form.method
    };

    // Append data as query params for GET request
    if (form.method.toUpperCase() === 'GET') {
        const queryParams = new URLSearchParams();
        for (const pair of formData) {
            queryParams.append(pair[0], pair[1]);
        }
        url += '?' + queryParams.toString();
    } else {
        // For POST request, add formData as body
        fetchOptions.body = formData;
    }

    fetch(url, fetchOptions)
    .then(response => response.json())
    .then(data => {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    })
    .catch(error => {
        document.getElementById('output').textContent = 'Error: ' + error;
    });
}
