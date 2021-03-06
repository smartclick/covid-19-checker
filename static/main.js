//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var chartContainer = document.getElementById("chart-container");

//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button
  console.log("submit");

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  // call the predict function of the backend
  predictImage(imageDisplay.src);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  chartContainer.innerHTML = "";
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  // show the preview of the image
  var fileName = encodeURI(file.name);
  if (file['type'].split('/')[0] == "image"){
    var reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onloadend = () => {
        imagePreview.src = URL.createObjectURL(file);

        show(imagePreview);
        hide(uploadCaption);

        // reset
        predResult.innerHTML = "";
        imageDisplay.classList.remove("loading");
        chartContainer.innerHTML = "";
        displayImage(reader.result, "image-display");
      };
  }
  else {
      let formData = new FormData();
      formData.append('file', file);
      fetch("/converter", {
        method: "POST",
        body: formData
      }).then(response => response.blob())
      .then(image => {
          if (image['type'].split('/')[0] == "image"){
            // Then create a local URL for that image and print it
              outside = URL.createObjectURL(image);
              file = image;
              var reader = new FileReader();
              reader.readAsDataURL(file);
              reader.onloadend = () => {
                imagePreview.src = URL.createObjectURL(file);

                show(imagePreview);
                hide(uploadCaption);

                // reset
                predResult.innerHTML = "";
                imageDisplay.classList.remove("loading");
                chartContainer.innerHTML = "";
                displayImage(reader.result, "image-display");
              };
          }else{
            window.alert("Oops! File format is wrong.");
          }

      })
  }
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  if (data.error){
    predResult.innerHTML = data.result + "<br><br>" + data.error;
  }else{
    predResult.innerHTML = data.result + "<br><br>Type: " + data.type + "<br>Confidence: " + data.probability;
  }
  if (data.condition_similarity_rate){
    chart = Highcharts.chart('chart-container', {
        chart: {
            plotBackgroundColor: null,
            plotBorderWidth: null,
            plotShadow: false,
            type: 'pie'
        },
        title: {text: 'Condition similarity rate'},
        tooltip: {pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'},
        accessibility: {
            point: {
                valueSuffix: '%'
            }
        },
        plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: 'pointer',
                dataLabels: {
                    enabled: true,
                    format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                    style: {
                        fontSize: 14
                    }
                }
            }
        },
        series: [{
            name: 'Brands',
            colorByPoint: true,
            data: data.condition_similarity_rate
        }]
    });
    }
    if (window.screen.width < 600)
    chart.setSize(window.screen.width-10);
  show(predResult);
  show(chartContainer);
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}