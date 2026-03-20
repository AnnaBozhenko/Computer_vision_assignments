from django.shortcuts import render
from app.services import filters, watermark_injection, sandbox_tools
from django.core.files.storage import FileSystemStorage

def home(request):
    return render(request, "home.html")


def process_image(request, op):
    if request.method == "GET":
        return render(request, "operation.html", {
        "operation": op
    })
    
    image = request.FILES['image']

    fs = FileSystemStorage(location="media/uploads")
    filename = fs.save(image.name, image)

    path = fs.path(filename)

    if op == "canny":
        sigma = float(request.POST["sigma"])
        low = float(request.POST["low_threshold"])
        high = float(request.POST["high_threshold"])
        result = filters.canny_edge(path, sigma=sigma, low_threshold=low, high_threshold=high)

    elif op == "sharpen":
        result = filters.sharpen_image(path)

    elif op == "median":
        kernel_size = int(request.POST["kernel_size"]) # added
        result = filters.median_filter(path, kernel_size)

    elif op == "morphology":
        dilation = bool(request.POST["dilation"]) # added
        kernel_size = int(request.POST["kernel_size"]) # added
        result = filters.morphology_filter(path, kernel_size, not dilation)

    elif op == "watermark_embed":
        watermark_img = request.FILES["watermark"]
        level = int(request.POST["level"])
        result = watermark_injection.inject_watermark(path, watermark_img, level)

    elif op == "watermark_extract":
        level = int(request.POST["level"])
        result = watermark_injection.extract_watermark(path, level)

    original_image = sandbox_tools.get_img(path)
    print(result)
    return render(request, "result.html", {
        "original": original_image,
        "result": result
    })
