function [Y, numDimsY] = onnxRoiAlign(X_, ONNXRois_, ONNXBatchIndices_, ONNXOutputHeight_, ONNXOutputWidth_, ONNXSamplingRatio_, ONNXSpatialScale_)
% Function to implement the ONNX RoiAlign operator
%#codegen

% All input tensors are in reverse-ONNX dimension ordering. Sizes are:
%
% X has reverse-ONNX dimension ordering: [w h c n]. Compared to MATLAB's
% convention, the spatial dimensions are transposed. So any MATLAB argument
% below that refers to x or y needs to use the other spatial dimension.
%
% ONNXRois: [4, num_rois], where each col is [x1; y1; x2; y2], where x and
% y refer to ONNX width and height dimensions respectively.
%
% ONNXBatchIndices: A 1D vector. In MATLAB, a column.
%
% The remaining 4 args are scalars.

% Copyright 2024 The MathWorks, Inc.
    X                       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    ONNXRois                = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXRois_);
    ONNXBatchIndices        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXBatchIndices_);
    ONNXOutputHeight        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXOutputHeight_);
    ONNXOutputWidth         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXOutputWidth_);
    ONNXSamplingRatio       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXSamplingRatio_);
    ONNXSpatialScale        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXSpatialScale_);
    
    OUTPUTSIZE = [ONNXOutputWidth, ONNXOutputHeight];                       % To get [height, width] in MATLAB, reverse the ONNX dims.
    ROISCALE = ONNXSpatialScale;
    
    MLstart_x = ONNXRois(2,:) + 1;                         	% Select the ONNX y1 row for the MATLAB x's. Make origin-1.
    MLstart_y = ONNXRois(1,:) + 1;                         	% Select the ONNX x1 row for the MATLAB y's. Make origin-1.
    MLend_x   = ONNXRois(4,:) + 1;                         	% Select the ONNX y2 row for the MATLAB x's. Make origin-1.
    MLend_y   = ONNXRois(3,:) + 1;                         	% Select the ONNX x2 row for the MATLAB y's. Make origin-1.
    
    if ONNXSamplingRatio <= 0
        SAMPLINGRATIO = "auto";
    else
        SAMPLINGRATIO = [ONNXSamplingRatio, ONNXSamplingRatio];
    end
    
    % Derive BOXES. IN MATLAB, boxes size should be A 5xN numeric matrix
    % holding N bounding boxes. Each bounding box is formatted as [start_x;
    % start_y; end_x; end_y; batchIdx].:
    BOXES  = [MLstart_x; MLstart_y; MLend_x; MLend_y; ONNXBatchIndices'+1];	% ONNXBatchIndices is a 1D ONNX tensor (a MATLAB column vector)
    
    % Call the dlarray function
    Y = roialign(dlarray(X,'SSCB'), BOXES, OUTPUTSIZE, 'ROIScale', ROISCALE, 'SamplingRatio', SAMPLINGRATIO);
    % Y is the RoI pooled output, 4-D tensor with reverse-ONNX dimension
    % ordering: (output_width, output_height, C, num_rois)
    numDimsY = 4;
end