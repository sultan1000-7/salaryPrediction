var region
var regionCount = 1;

var modal = document.getElementById("myModal");
var span = document.getElementsByClassName("close")[0];

var currentMethod = 'Линейная регрессия';


span.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}


document.getElementById('linear').addEventListener('click', function() {
    document.getElementById('currentMethod').textContent = 'Текущий метод: Линейная регрессия';
    currentMethod = 'Линейная регрессия';
});
document.getElementById('boosting').addEventListener('click', function() {
    document.getElementById('currentMethod').textContent = 'Текущий метод: Бустинг';
    currentMethod = 'Бустинг';
});
document.getElementById('knn').addEventListener('click', function() {
    document.getElementById('currentMethod').textContent = 'Текущий метод: Метод k-ближайших соседей (k-NN)';
    currentMethod = 'Метод k-ближайших соседей (k-NN)';
});
document.getElementById('svm').addEventListener('click', function() {
    document.getElementById('currentMethod').textContent = 'Текущий метод: Метод опорных векторов (SVM)';
    currentMethod = 'Метод опорных векторов (SVM)';
});

function searchFunction() {
    var input, filter, ul, li, a, i, txtValue;
    input = document.getElementById('search');
    filter = input.value.toUpperCase();
    ul = document.getElementById("regionList");
    li = ul.getElementsByTagName('li');

    for (i = 0; i < li.length; i++) {
        txtValue = li[i].textContent || li[i].innerText;

        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            li[i].style.display = "";
        } else {
            li[i].style.display = "none";
        }
    }
}
function selectRegion(element) {
    region.value = element.textContent;
    modal.style.display = "none";
}

function toggleForecast() {
    var profession = document.getElementById("inputProfession");
    var methods = document.getElementById("methods")
    if (profession.style.display === "none") {
        profession.style.display = "block";
        methods.style.display = "flex";
        
    } else {
        profession.style.display = "none";
        methods.style.display = "none";
    }
}
function openModal(input) {
    modal.style.display = "block";
    region = document.getElementById(input.id)
    
}

function addRegionInput() {
    var newInput = document.createElement("input");
    newInput.type = "text";
    newInput.name = "region";
    newInput.className = "other-input";
    newInput.placeholder = "Регион";
    newInput.onclick = function() { openModal(this); };
    document.getElementById('regionContainer').appendChild(newInput);
}

function sendData() {
    var profession = document.getElementById('profession').value;
    var experience = document.getElementById('experience').value;
    var employment = document.getElementById('employment').value;
    var schedule = document.getElementById('schedule').value;
    var regions = [];
    for (var i = 1; i <= 3; i++) {
        let text = document.getElementById('region' + i).value
        if (text !== "" && text !== undefined && text !== null){
            regions.push(text)
        }
    }
    
    fetch('/data_for_forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'profession': profession,
            'experience': experience,
            'employment': employment,
            'schedule': schedule,
            'regions': JSON.stringify(regions),
            'method': currentMethod
        })
    })
    .then(response => response.json())
    .then(response => {
        if (response.status === 'ok') {
            fetch('/get_data_for_forecast', {
                method: 'GET'
            })
            .then(response => response.json())
            .then(data => {
                // Записываем данные в div с id "dataPlace"
                var rmse = data.rmse;
                var y_pred = data.y_pred;
                var rmseElement = document.getElementById('rmse');
                rmseElement.innerHTML = 'RMSE: ' + rmse.toFixed(2);
                // Вычисляем значение цвета в зависимости от rmse
                var greenComponent = Math.floor((1 - rmse) * 255);
                var redComponent = Math.floor(rmse * 255);
                rmseElement.style.color = 'rgb(' + redComponent + ',' + greenComponent + ',0)';
                var size = Math.floor(y_pred.length / 3);
                var y_pred1 = y_pred.slice(0, size);
                var y_pred2 = y_pred.slice(size, 2*size);
                var y_pred3 = y_pred.slice(2*size, y_pred.length);
                var y_pred4 = y_pred.slice(0, y_pred.length);
                
                // Вычисляем среднее значение каждого массива
                var avg1 = y_pred1.reduce((a, b) => a + b, 0) / y_pred1.length;
                var avg2 = y_pred2.reduce((a, b) => a + b, 0) / y_pred2.length;
                var avg3 = y_pred3.reduce((a, b) => a + b, 0) / y_pred3.length;
                var avg4 = y_pred4.reduce((a, b) => a + b, 0) / y_pred4.length;
                
                // Вставляем средние значения в элемент с id "y_pred"
                document.getElementById('y_pred').innerHTML = 
                    'Предсказанные значения:<br>' + avg1.toFixed(2) + '<br>' + avg2.toFixed(2) + '<br>' + avg3.toFixed(2) + '<br>' + avg4.toFixed(2);

                var imageUrl = '/get_histogram/?t=' + new Date().getTime();
                document.getElementById('imageContainer1').innerHTML = '<img src="' + imageUrl + '" alt="Histogram">';

                var imageUrl = '/get_predicted_histogram/?t=' + new Date().getTime();
                document.getElementById('imageContainer2').innerHTML = '<img src="' + imageUrl + '" alt="Histogram">';

                document.getElementById('dataPlace').style.display = 'flex';
            });
        }
    });
}