document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("digitCanvas");
    const ctx = canvas.getContext("2d");
    const clearBtn = document.getElementById("clearBtn");
    const predictBtn = document.getElementById("predictBtn");
    const predictionText = document.getElementById("predictionText");

    let drawing = false;

    // 1. Initialize canvas background to black
    function clearCanvas() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (predictionText) predictionText.textContent = "Prediction: ";
    }
    clearCanvas();

    // 2. Stroke configuration (white ink for MNIST digits)
    ctx.strokeStyle = "white";
    ctx.lineWidth = 3.3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // 3. Helper to get scaled coordinates (280px CSS -> 28px canvas buffer)
    function getCanvasCoords(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    // 4. Drawing event handlers
    function startDrawing(event) {
        drawing = true;
        const { x, y } = getCanvasCoords(event);
        ctx.beginPath();
        ctx.moveTo(x, y);
        // Draw a single dot even if the user just clicks without moving
        ctx.lineTo(x, y);
        ctx.stroke();
    }

    function draw(event) {
        if (!drawing) return;
        const { x, y } = getCanvasCoords(event);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function stopDrawing() {
        if (!drawing) return;
        drawing = false;
        ctx.beginPath();
    }

    // Canvas listeners
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener("click", clearCanvas);
    }

    // Predict button
    if (predictBtn) {
        predictBtn.addEventListener("click", () => {
            const image = canvas.toDataURL("image/png");

            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ image: image })
            })
            .then(res => res.json())
            .then(data => {
                console.log("Response from server:", data);
                const predictionText = document.getElementById("predictionText");
                console.log("Found predictionText element?", predictionText);
                
                if (predictionText) {
                    predictionText.textContent = "Prediction: " + data.prediction;
                }
            })
            .catch(err => console.error("Prediction failed:", err));
        });
    }
});