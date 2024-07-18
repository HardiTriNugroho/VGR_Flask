// Reset previous prediction and accuracy when a new file is selected
document.getElementById("audioFile").addEventListener("change", function () {
  document.getElementById("response").innerHTML = "";
  document.getElementById("accuracyValue").innerHTML = "";
});

async function uploadFile() {
  var loadingDiv = document.getElementById("loading");
  const fileInput = document.getElementById("audioFile");
  if (!fileInput.files.length) {
    alert("Please select a file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  loadingDiv.style.display = "block";

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  loadingDiv.style.display = "none";
  if (response.ok) {
    document.getElementById("response").innerHTML = `${result.gender}`;
    document.getElementById(
      "accuracyValue"
    ).innerHTML = `${result.accuracy.toFixed(2)}%`;
  } else {
    document.getElementById("response").innerHTML = `Error: ${result.error}`;
  }

  // Clear the input file after the upload button is pressed
  fileInput.value = "";
}

// #DROP FILE
const dropContainer = document.getElementById("dropcontainer");
const fileInput = document.getElementById("audioFile");

dropContainer.addEventListener(
  "dragover",
  (e) => {
    // prevent default to allow drop
    e.preventDefault();
  },
  false
);

dropContainer.addEventListener("dragenter", () => {
  dropContainer.classList.add("drag-active");
});

dropContainer.addEventListener("dragleave", () => {
  dropContainer.classList.remove("drag-active");
});

dropContainer.addEventListener("drop", (e) => {
  e.preventDefault();
  dropContainer.classList.remove("drag-active");
  fileInput.files = e.dataTransfer.files;
});
