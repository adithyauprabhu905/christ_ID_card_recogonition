# christ_ID_card_recogonition
This MATLAB project verifies the authenticity of ID cards by detecting the presence of the keyword "CHRIST" (as text or logo) using image processing and OCR techniques. It supports both webcam capture and manual image upload, making it versatile for real-time or offline verification.


# CHRIST ID Card Recognition using MATLAB

This MATLAB project verifies the authenticity of ID cards by detecting the presence of the keyword **"CHRIST"** (as text or logo) using image processing and OCR techniques. It supports both **webcam capture** and **manual image upload**, making it versatile for real-time or offline verification.

## 🔍 Features

- 📷 **Webcam or File Input**: Choose to capture the ID card live or upload an image.
- 🧠 **Automatic & Manual Card Detection**: Smart detection of the ID card from the full image with an option for manual cropping.
- 📦 **Region Analysis**: Extracts key regions (header, logo area, footer, full card) for targeted OCR.
- 💡 **Multi-step Image Processing**: Applies various filters (contrast enhancement, sharpening, edge detection, etc.) for improved text recognition.
- 🧾 **OCR and Text Matching**: Detects the presence of "CHRIST" or its variants using MATLAB’s OCR engine.
- ✅ **Approval Decision**: Verifies the ID card based on OCR confidence and region.
- 🖼️ **Visual Feedback**: Displays processed regions and results in a user-friendly format.
- 📁 **Report Generation**: Saves screenshots and a detailed text report of the verification.

## 📂 Output

The script returns:
- `is_approved`: Boolean status indicating approval.
- `verification_data`: Struct with confidence score, processed image, detected text, and region info.

## 🛠️ Technologies Used

- MATLAB
- Image Processing Toolbox
- OCR Toolbox

## 🧪 Example Usage

```matlab
[approved, data] = christ_ID_card_recogonition(true);   % Use webcam
[approved, data] = christ_ID_card_recogonition(false);  % Upload an image
