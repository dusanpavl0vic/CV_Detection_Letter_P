from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def generate_scales_and_angles():
    # kod je spor zbog ovog ali morao sam da prodjem kroz sve velicine slika i sve rotacije
    scales = [round(s, 2) for s in np.arange(0.5, 2.1, 0.1)]
    angles = list(range(0, 360, 10))
    return scales, angles

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

def match_templates(image_path, template_paths, scales, angles):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = []

    for template_path in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        for scale in scales:
            try:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                continue

            for angle in angles:
                rotated_template = rotate_image(scaled_template, angle)
                
                result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > 0.7:
                    h, w = rotated_template.shape
                    x, y = max_loc
                    detected_region = image[y:y+h, x:x+w]

                    if detected_region.shape == rotated_template.shape:
                        # koristim ssim za proveru proveru izmedju detektovanih regiona
                        #  i sablona koji su rotirani i uvecani i umanjeni
                        similarity = ssim(rotated_template, detected_region)
                        if similarity > 0.7:
                            results.append({
                                'template': template_path,
                                'scale': scale,
                                'angle': angle,
                                'location': max_loc,
                                'score': max_val,
                                'similarity': similarity,
                                'shape': rotated_template.shape
                            })
    return results

def detect_and_show_p(image_path, template_paths):
    scales, angles = generate_scales_and_angles()
    results = match_templates(image_path, template_paths, scales, angles)
    image = cv2.imread(image_path)

    for result in results:
        top_left = result['location']
        h, w = result['shape']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2)

    cv2.imwrite("out_img.png", image)

    cv2.imshow('Detected Letters', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'importcv2.png'
template_paths = ['2.png', '3.png']

detect_and_show_p(image_path, template_paths)
