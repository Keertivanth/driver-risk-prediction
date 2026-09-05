function setLoading(isLoading) {
    let status = document.getElementById("status");

    if (isLoading) {
        status.innerText = "⏳ Predicting...";
    } else {
        status.innerText = "";
    }
}

function animateBar(value) {
    let bar = document.getElementById("progress");
    bar.style.width = "0%";

    let width = 0;
    let interval = setInterval(() => {
        if (width >= value) {
            clearInterval(interval);
        } else {
            width++;
            bar.style.width = width + "%";
        }
    }, 10);
}

function updateUI(data) {
    document.getElementById("type").innerText =
        "Driving Type: " + data.type;

    document.getElementById("risk").innerText =
        "Risk Score: " + data.risk + "%";

    let bar = document.getElementById("progress");

    if (data.risk < 50) {
        bar.style.background = "green";
    } else {
        bar.style.background = "red";
    }

    animateBar(data.risk);
}

function uploadFile() {
    let fileInput = document.getElementById("fileInput");
    let status = document.getElementById("uploadStatus");

    if (fileInput.files.length === 0) {
        alert("Please select a CSV file");
        return;
    }

    status.innerText = "Uploading...";

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    setLoading(true);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        setLoading(false);
        status.innerText = "Done ✅";

        if (data.error) {
            alert(data.error);
        } else {
            updateUI(data);
        }
    })
    .catch(err => {
        setLoading(false);
        status.innerText = "Error ❌";
        console.error(err);
    });
}

function useSample() {
    setLoading(true);

    fetch("/sample")
    .then(res => res.json())
    .then(data => {
        setLoading(false);
        updateUI(data);
    })
    .catch(err => {
        setLoading(false);
        console.error(err);
    });
}