function [is_approved, verification_data] = christ_id_verification(use_webcam)
% CHRIST_ID_VERIFICATION - Verifies if an ID card contains "CHRIST" text/logo
%
% Syntax:
%   [is_approved, verification_data] = christ_id_verification()
%   [is_approved, verification_data] = christ_id_verification(use_webcam)
%
% Inputs:
%   use_webcam - Optional boolean parameter:
%               true (default) - Use webcam for image capture
%               false - Prompt for file selection
%
% Outputs:
%   is_approved - Boolean indicating if CHRIST was detected on the ID card
%   verification_data - Struct containing detailed verification information:
%       .approval_roi - Region where CHRIST was detected
%       .approval_source - Processing method that detected CHRIST
%       .approval_text - Text containing CHRIST that was detected
%       .confidence_score - Confidence level of detection (0-1)
%       .cropped_card - The cropped ID card image
%       .best_processed_img - The processed image that yielded detection
%       .regions - Struct with different analyzed regions of the card
%
% Examples:
%   % Use with webcam (default)
%   [approved, data] = christ_id_verification()
%
%   % Explicitly use webcam
%   [approved, data] = christ_id_verification(true)
%
%   % Use file upload instead of webcam
%   [approved, data] = christ_id_verification(false)

    % Set default to use webcam if not specified
    if nargin < 1
        use_webcam = true;
    end

    % Initialize output struct
    verification_data = struct();
    verification_data.approval_roi = "";
    verification_data.approval_source = "";
    verification_data.approval_text = "";
    verification_data.confidence_score = 0;
    verification_data.cropped_card = [];
    verification_data.best_processed_img = [];
    verification_data.regions = struct();
    
    % Get input image using webcam or file upload
    img = get_input_image(use_webcam);
    if isempty(img)
        is_approved = false;
        return;
    end
    
    % Detect and crop ID card
    [id_card, crop_position] = detect_id_card(img);
    if isempty(id_card)
        is_approved = false;
        return;
    end
    
    % Store cropped card in output
    verification_data.cropped_card = id_card;
    
    % Extract regions of interest from the ID card
    [rois, roi_names] = extract_card_regions(id_card);
    
    % Store regions in output
    for r = 1:length(rois)
        field_name = strrep(lower(roi_names{r}), ' ', '_');
        verification_data.regions.(field_name) = rois{r};
    end
    
    % Process regions and detect "CHRIST"
    [is_approved, approval_data] = process_and_detect(rois, roi_names);
    
    % Update output struct with results
    if is_approved
        verification_data.approval_roi = approval_data.roi;
        verification_data.approval_source = approval_data.source;
        verification_data.approval_text = approval_data.text;
        verification_data.confidence_score = approval_data.confidence;
        verification_data.best_processed_img = approval_data.processed_img;
    end
    
    % Optional: Generate report if requested
    if nargout < 2 % No output requested, display results
        display_results(img, id_card, rois, roi_names, is_approved, verification_data);
    end
end

function img = get_input_image(use_webcam)
    % Get image from webcam or file based on parameter
    
    % Try to use webcam if requested
    if use_webcam
        try
            % Initialize webcam
            cam = webcam();
            
            % Create a preview window with instructions
            figure('Name', 'Camera Preview', 'NumberTitle', 'off', 'Position', [100, 100, 640, 480]);
            preview_ax = axes;
            
            % Display instructions
            instructions = uicontrol('Style', 'text', ...
                           'String', 'Position the ID card clearly in frame and press SPACE to capture', ...
                           'Position', [20, 20, 600, 30], ...
                           'BackgroundColor', [0.9, 0.9, 0.9]);
            
            % Setup preview
            preview(cam, preview_ax);
            
            % Wait for user to press space to capture
            fig = gcf;
            set(fig, 'KeyPressFcn', @(obj, evt) assignin('base', 'key_pressed', evt.Key));
            
            key_pressed = '';
            while ~strcmp(key_pressed, 'space')
                pause(0.1);
                try
                    key_pressed = evalin('base', 'key_pressed');
                catch
                    key_pressed = '';
                end
            end
            
            % Capture image
            img = snapshot(cam);
            
            % Clean up variables and camera
            evalin('base', 'clear key_pressed');
            clear cam;
            close;
            
            % Display captured image for confirmation
            figure('Name', 'Captured Image', 'NumberTitle', 'off', 'Position', [100, 100, 640, 520]);
            imshow(img);
            title('Captured Image');
            
            % Add confirmation buttons
            uicontrol('Style', 'pushbutton', 'String', 'Use This Image', ...
                     'Position', [180, 20, 120, 30], ...
                     'Callback', @(src, event) assignin('base', 'user_decision', 1));
            
            uicontrol('Style', 'pushbutton', 'String', 'Retake Image', ...
                     'Position', [340, 20, 120, 30], ...
                     'Callback', @(src, event) assignin('base', 'user_decision', 0));
            
            % Wait for user decision
            uiwait;
            try
                user_decision = evalin('base', 'user_decision');
                evalin('base', 'clear user_decision');
                if user_decision == 0
                    close;
                    % Recursively call for retake
                    img = get_input_image(use_webcam);
                    return;
                end
            catch
                % Default to accepting the image if window was closed
                user_decision = 1;
            end
            close;
            
            % Check if we got a valid image
            if isempty(img)
                % If webcam capture failed, fall back to file selection
                choice = questdlg('Webcam capture failed. Would you like to upload an image instead?', ...
                                 'Camera Error', 'Yes', 'Cancel', 'Yes');
                if strcmp(choice, 'Yes')
                    img = get_input_image(false);
                else
                    img = [];
                    return;
                end
            end
        catch webcam_error
            % Handle webcam errors
            warning('Webcam error: %s', webcam_error.message);
            choice = questdlg('Webcam not available or error occurred. Would you like to upload an image instead?', ...
                             'Camera Error', 'Yes', 'Cancel', 'Yes');
            if strcmp(choice, 'Yes')
                img = get_input_image(false);
            else
                img = [];
                return;
            end
        end
    else
        % File selection if webcam not requested
        [fileName, pathName] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image Files'}, 'Select an ID card image');
        if fileName == 0
            disp('Operation cancelled');
            img = [];
            return;
        end
        try
            img = imread(fullfile(pathName, fileName));
        catch
            warndlg('Could not read the selected file. Please select a valid image file.', 'File Error');
            img = [];
        end
    end
end

function [id_card, crop_position] = detect_id_card(img)
    % Auto-detect ID card or allow manual selection
    
    % Initialize default output
    id_card = [];
    crop_position = [];
    
    % ID card-specific preprocessing for detection
    gray_img = rgb2gray(img);
    edges = edge(gray_img, 'Canny');
    
    % Use morphology to close gaps in the edges
    se = strel('line', 20, 0);
    edges_h = imclose(edges, se);
    se = strel('line', 20, 90);
    edges_v = imclose(edges, se);
    edges_combined = edges_h | edges_v;
    
    % Find rectangles that could be ID cards
    filled = imfill(edges_combined, 'holes');
    area_filtered = bwareaopen(filled, 10000); % Remove small objects
    
    % Get properties of potential card regions
    stats = regionprops(area_filtered, 'BoundingBox', 'Area', 'Extent');
    
    % Auto-detected card region
    auto_detected = false;
    auto_box = [];
    
    % Look for rectangular regions with appropriate aspect ratio for ID cards
    if ~isempty(stats)
        for i = 1:length(stats)
            bbox = stats(i).BoundingBox;
            aspect = bbox(3) / bbox(4);
            extent = stats(i).Extent; % How rectangular is the region
            
            % Check if region has ID card-like properties
            % Typical ID cards have width/height ratio between 1.4 and 1.7
            % and are quite rectangular (extent close to 1)
            if aspect > 1.4 && aspect < 1.7 && extent > 0.7
                auto_box = bbox;
                auto_detected = true;
                break;
            end
        end
    end
    
    % If auto-detection found a potential card
    if auto_detected
        % Display auto-detected card for user confirmation
        fig = figure('Name', 'ID Card Detection', 'NumberTitle', 'off', 'Position', [100, 100, 640, 520]);
        imshow(img);
        hold on;
        rectangle('Position', auto_box, 'EdgeColor', 'g', 'LineWidth', 2);
        title('Auto-detected ID card');
        hold off;
        
        % Add confirmation buttons
        uicontrol('Style', 'pushbutton', 'String', 'Accept This Selection', ...
                 'Position', [160, 20, 150, 30], ...
                 'Callback', @(src, event) assignin('base', 'auto_accept', 1));
        
        uicontrol('Style', 'pushbutton', 'String', 'Draw Manual Selection', ...
                 'Position', [330, 20, 150, 30], ...
                 'Callback', @(src, event) assignin('base', 'auto_accept', 0));
        
        % Wait for user decision
        uiwait;
        try
            auto_accept = evalin('base', 'auto_accept');
            evalin('base', 'clear auto_accept');
            if auto_accept == 1
                crop_position = auto_box;
            else
                % User rejected auto-detection, let them draw manually
                close(fig);
                crop_position = manual_crop_selection(img);
                return;
            end
        catch
            % Default to accepting if window was closed
            crop_position = auto_box;
        end
        close(fig);
    else
        % No auto-detection, let user crop manually
        crop_position = manual_crop_selection(img);
    end
    
    % If no valid crop position, return empty
    if isempty(crop_position)
        return;
    end
    
    % Crop the image based on selection
    try
        x = max(1, round(crop_position(1)));
        y = max(1, round(crop_position(2)));
        w = round(crop_position(3));
        h = round(crop_position(4));
        
        % Ensure coordinates are within image bounds
        [height, width, ~] = size(img);
        w = min(w, width-x+1);
        h = min(h, height-y+1);
        
        % Crop the user-selected region
        id_card = imcrop(img, [x, y, w, h]);
    catch
        warning('Failed to crop image.');
        id_card = [];
    end
end

function crop_position = manual_crop_selection(img)
    % Allow user to manually select the ID card region
    fig = figure('Name', 'Manual Crop', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
    imshow(img);
    title('Draw a rectangle around the ID card, then double-click inside it');
    
    % Instructions text
    uicontrol('Style', 'text', ...
               'String', 'Draw a rectangle precisely around the edges of the ID card', ...
               'Position', [200, 20, 400, 30], ...
               'BackgroundColor', [0.9, 0.9, 0.9]);
    
    try
        cropped_roi = drawrectangle('Label', 'Select ID Card');
        wait(cropped_roi);
        crop_position = cropped_roi.Position; % [x, y, width, height]
        close(fig);
    catch
        warning('Manual selection failed or was cancelled.');
        crop_position = [];
        close(fig);
    end
end

function [rois, roi_names] = extract_card_regions(id_card)
    % Extract different regions of interest from the ID card
    [card_h, card_w, ~] = size(id_card);
    
    % Define regions of interest (ROIs) where "CHRIST" is likely to appear
    % Header region (top 25% where institution name often appears)
    header_roi = imcrop(id_card, [0, 0, card_w, card_h*0.25]);
    
    % Logo region (top-left corner where logos often appear)
    logo_roi = imcrop(id_card, [0, 0, card_w*0.33, card_h*0.33]);
    
    % Footer region (bottom 20% where additional institution info might appear)
    footer_roi = imcrop(id_card, [0, card_h*0.8, card_w, card_h*0.2]);
    
    % Entire card as well (some IDs might have watermarks or background text)
    whole_card_roi = id_card;
    
    % List of ROIs and their descriptions
    rois = {header_roi, logo_roi, footer_roi, whole_card_roi};
    roi_names = {'Header', 'Logo Area', 'Footer', 'Whole Card'};
end

function [is_approved, approval_data] = process_and_detect(rois, roi_names)
    % Initialize output
    is_approved = false;
    approval_data = struct();
    approval_data.roi = "";
    approval_data.source = "";
    approval_data.text = "";
    approval_data.confidence = 0;
    approval_data.processed_img = [];
    
    % Show progress indicator
    progress_dlg = waitbar(0, 'Analyzing ID card regions...', 'Name', 'CHRIST ID Verification');
    total_steps = length(rois) * 13; % 13 processing methods per ROI
    current_step = 0;
    
    % Process each ROI with multiple enhancement techniques
    for r = 1:length(rois)
        roi = rois{r};
        roi_name = roi_names{r};
        
        waitbar(current_step/total_steps, progress_dlg, sprintf('Analyzing %s region...', roi_name));
        
        % Generate processed versions of the current ROI
        [processed_rois, processing_names] = process_roi(roi);
        
        % Run OCR on all processed versions of the ROI
        for p = 1:length(processed_rois)
            current_step = current_step + 1;
            waitbar(current_step/total_steps, progress_dlg, ...
                    sprintf('Analyzing %s region with %s...', roi_name, processing_names{p}));
            
            % Perform OCR
            ocr_results = ocr(processed_rois{p});
            
            % Check for CHRIST in OCR text
            [christ_found, result_text, conf] = check_for_christ(ocr_results);
            
            if christ_found && conf > approval_data.confidence
                is_approved = true;
                approval_data.roi = roi_name;
                approval_data.source = processing_names{p};
                approval_data.text = result_text;
                approval_data.confidence = conf;
                approval_data.processed_img = processed_rois{p};
            end
        end
    end
    
    close(progress_dlg);
end

function [processed_rois, processing_names] = process_roi(roi)
    % Apply various image processing techniques to the ROI
    
    % 1. Basic grayscale conversion
    gray_roi = rgb2gray(roi);
    
    % 2. Contrast enhancement with different parameters
    enhanced_roi = imadjust(gray_roi, [0.3 0.7], [0 1]);  % More aggressive contrast
    
    % 3. Sharpening to enhance text edges
    sharpened_roi = imsharpen(gray_roi, 'Radius', 1.5, 'Amount', 1.2);
    
    % 4. Color thresholding to isolate potential text in common ID colors
    dark_mask = gray_roi < 100;
    
    % 5. Edge detection to enhance text boundaries
    edges_roi = edge(gray_roi, 'Canny');
    dilated_edges = imdilate(edges_roi, strel('disk', 1));
    
    % 6. CLAHE for better local contrast
    clahe_roi = adapthisteq(gray_roi, 'ClipLimit', 0.02);
    
    % 7. Denoising
    denoised_roi = wiener2(gray_roi, [3 3]);
    
    % 8. Color channels separation - check each channel (for color logos)
    if size(roi, 3) == 3
        red_channel = roi(:,:,1);
        green_channel = roi(:,:,2);
        blue_channel = roi(:,:,3);
    else
        red_channel = gray_roi;
        green_channel = gray_roi;
        blue_channel = gray_roi;
    end
    
    % 9. Binarization with different thresholds
    binary_roi1 = imbinarize(gray_roi);
    binary_roi2 = imbinarize(gray_roi, 'adaptive', 'Sensitivity', 0.6);
    binary_roi3 = imbinarize(clahe_roi, 'adaptive', 'Sensitivity', 0.4);
    
    % Combine all processed images for this ROI
    processed_rois = {gray_roi, enhanced_roi, sharpened_roi, dark_mask, ...
                    dilated_edges, clahe_roi, denoised_roi, red_channel, ...
                    green_channel, blue_channel, binary_roi1, binary_roi2, binary_roi3};
    
    processing_names = {'Grayscale', 'Enhanced', 'Sharpened', 'Dark Mask', ...
                       'Edge Detection', 'CLAHE', 'Denoised', 'Red Channel', ...
                       'Green Channel', 'Blue Channel', 'Binary', 'Adaptive Binary', ...
                       'CLAHE Binary'};
end

function [contains_christ, result_text, confidence] = check_for_christ(ocr_result)
    % Check if OCR results contain "CHRIST" text
    contains_christ = false;
    result_text = "";
    confidence = 0;
    
    if ~isempty(ocr_result.Text)
        result_text = ocr_result.Text;
        
        % Look for variations of "CHRIST" to account for OCR errors
        variations = {'CHRIST', 'CHRlST', 'CHRlST', 'CHRlS', 'CHRIS', 'CHRI5T', 'CHRI$T', 'CHRlST', 'CHRISTI', 'CHRISTIA'};
        
        for i = 1:length(variations)
            if contains(upper(ocr_result.Text), variations{i})
                contains_christ = true;
                
                % Find the matching word in the recognized words
                for j = 1:length(ocr_result.Words)
                    if contains(upper(ocr_result.Words{j}), variations{i})
                        % Get confidence for this word
                        confidence = ocr_result.WordConfidences(j);
                        break;
                    end
                end
                
                if confidence > 0
                    break;  % Found a match with confidence
                end
            end
        end
        
        % If no direct word match but text contains CHRIST, use average confidence
        if contains_christ && confidence == 0 && ~isempty(ocr_result.WordConfidences)
            confidence = mean(ocr_result.WordConfidences);
        end
    end
end

function display_results(img, id_card, rois, roi_names, is_approved, verification_data)
    % Create visualization of verification results
    figure('Name', 'CHRIST ID Card Verification', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);
    
    % Display original image
    subplot(3, 3, 1);
    imshow(img);
    title('Original Image');
    
    % Display cropped ID card
    subplot(3, 3, 2);
    imshow(id_card);
    title('Cropped ID Card');
    
    % Display regions
    subplot(3, 3, 3);
    imshow(rois{1});  % Header
    title(roi_names{1});
    
    subplot(3, 3, 4);
    imshow(rois{2});  % Logo
    title(roi_names{2});
    
    subplot(3, 3, 5);
    imshow(rois{3});  % Footer
    title(roi_names{3});
    
    % Display best processed image if found
    subplot(3, 3, 6);
    if ~isempty(verification_data.best_processed_img)
        imshow(verification_data.best_processed_img);
        title(['Best Processing: ' verification_data.approval_source]);
    else
        imshow(rgb2gray(id_card));
        title('Processed Image');
    end
    
    % Display verification result
    subplot(3, 3, [7,8,9]);
    imshow(id_card);
    hold on;
    
    % Add approval/disapproval message
    if is_approved
        % Display approval message in green
        text_color = 'green';
        message = 'ID CARD APPROVED';
        detailed_msg = sprintf('"CHRIST" detected in %s region using %s (Confidence: %.2f%%)', ...
            verification_data.approval_roi, verification_data.approval_source, ...
            verification_data.confidence_score*100);
    else
        % Display disapproval message in red
        text_color = 'red';
        message = 'ID CARD NOT APPROVED';
        detailed_msg = '"CHRIST" not detected in ID card';
    end
    
    % Get dimensions of ID card for text positioning
    [card_h, card_w, ~] = size(id_card);
    
    % Add messages to the image
    text(card_w*0.1, card_h*0.4, message, 'Color', text_color, 'FontSize', 14, 'FontWeight', 'bold');
    text(card_w*0.1, card_h*0.5, detailed_msg, 'Color', text_color, 'FontSize', 10);
    
    % If approved, show the detected text
    if is_approved
        text(card_w*0.1, card_h*0.6, 'Extracted text:', 'Color', text_color, 'FontSize', 10);
        text(card_w*0.1, card_h*0.7, verification_data.approval_text, 'Color', text_color, 'FontSize', 8);
    end
    
    hold off;
    title('Verification Result');
    
    % Print result to command window
    if is_approved
        fprintf('\n=================================\n');
        fprintf('ID CARD VERIFICATION: APPROVED ✓\n');
        fprintf('=================================\n');
        fprintf('Reason: "CHRIST" detected on ID card\n');
        fprintf('Location: %s region\n', verification_data.approval_roi);
        fprintf('Detection method: %s\n', verification_data.approval_source);
        fprintf('Confidence score: %.2f%%\n', verification_data.confidence_score*100);
        fprintf('Extracted text: %s\n', verification_data.approval_text);
        fprintf('=================================\n');
    else
        fprintf('\n=================================\n');
        fprintf('ID CARD VERIFICATION: NOT APPROVED ✗\n');
        fprintf('=================================\n');
        fprintf('Reason: "CHRIST" not detected on ID card\n');
        fprintf('=================================\n');
    end
    
    % Save verification report if user wants to
    choice = questdlg('Would you like to save a verification report?', 'Save Report', 'Yes', 'No', 'Yes');
    if strcmp(choice, 'Yes')
        export_verification_report(id_card, is_approved, verification_data);
    end
end

function export_verification_report(id_card, is_approved, verification_data)
    % Export verification results to files
    try
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        result_filename = ['christ_id_verification_', timestamp];
        
        % Save figure
        saveas(gcf, [result_filename, '.png']);
        
        % Save ID card image with verification result
        verification_img = id_card;
        [result_h, result_w, ~] = size(verification_img);
        
        % Create a result image with text overlay
        result_fig = figure('Visible', 'off');
        imshow(verification_img);
        hold on;
        
        if is_approved
            rectangle('Position', [10, 10, result_w-20, result_h-20], 'EdgeColor', 'green', 'LineWidth', 3);
            text(result_w*0.1, result_h*0.1, 'APPROVED', 'Color', 'green', 'FontSize', 24, 'FontWeight', 'bold');
            text(result_w*0.1, result_h*0.9, sprintf('CHRIST detected (%.1f%%)', verification_data.confidence_score*100), ...
                'Color', 'green', 'FontSize', 14);
        else
            rectangle('Position', [10, 10, result_w-20, result_h-20], 'EdgeColor', 'red', 'LineWidth', 3);
            text(result_w*0.1, result_h*0.1, 'NOT APPROVED', 'Color', 'red', 'FontSize', 24, 'FontWeight', 'bold');
            text(result_w*0.1, result_h*0.9, 'CHRIST not detected', 'Color', 'red', 'FontSize', 14);
        end
        hold off;
        
        saveas(result_fig, [result_filename, '_result.png']);
        close(result_fig);
        
        % Save detailed text report         fid = fopen([result_filename, '.txt'], 'w');
        fprintf(fid, 'CHRIST ID Card Verification Report\n');
        fprintf(fid, 'Date/Time: %s\n\n', datestr(now));
        
        if is_approved
            fprintf(fid, 'RESULT: APPROVED\n');
            fprintf(fid, 'Location: %s region\n', verification_data.approval_roi);
            fprintf(fid, 'Detection Method: %s\n', verification_data.approval_source);
            fprintf(fid, 'Confidence: %.2f%%\n', verification_data.confidence_score*100);
            fprintf(fid, 'Extracted Text: %s\n', verification_data.approval_text);
        else
            fprintf(fid, 'RESULT: NOT APPROVED\n');
            fprintf(fid, 'Reason: CHRIST logo/text not detected\n');
        end
        fclose(fid);
        
        % Confirmation message
        msgbox(sprintf('Report saved as:\n- %s.png\n- %s_result.png\n- %s.txt', ...
                      result_filename, result_filename, result_filename), 'Report Saved');
    catch err
        warndlg(sprintf('Could not save verification results to file: %s', err.message), 'Save Error');
    end
end