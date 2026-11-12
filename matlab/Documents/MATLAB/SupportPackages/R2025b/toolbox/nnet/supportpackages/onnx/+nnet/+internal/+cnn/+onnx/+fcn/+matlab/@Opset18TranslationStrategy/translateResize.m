function nodeTranslation = translateResize(~, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024-2026 The MathWorks, Inc. 

    % Resize-11 adds 5 attributes and 2 inputs.
    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "antialias"                         "INT"       true    0
        "axes"                              "INTS"      true    []
        "coordinate_transformation_mode"    "STRING"    true    "half_pixel"
        "cubic_coeff_a"                     "FLOAT"     true    -0.75
        "exclude_outside"                   "INT"       true    0
        "extrapolation_value"               "FLOAT"     true    0.0
        "keep_aspect_ratio_policy"          "STRING"    true    "stretch"
        "mode"                              "STRING"    true    "nearest"
        "nearest_mode"                      "STRING"    true    "round_prefer_floor"
        });
    % Parse the attributes
    [antialias, axes, coordinate_transformation_mode,cubic_coeff_a, exclude_outside, ...
        extrapolation_value, keep_aspect_ratio_policy, mode, nearest_mode] ...
        = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    nodeTranslation.Issues = [nodeTranslation.Issues
        iValidateInput(node, mode, cubic_coeff_a, coordinate_transformation_mode, exclude_outside, extrapolation_value, nearest_mode)];
    if ~isempty(nodeTranslation.Issues)
        return;
    end
    % if the coordinate_transformation_mode is "pytorch_half_pixel", set
    % it to "half_pixel" and throw out a warning
    if isequal(coordinate_transformation_mode, "pytorch_half_pixel")
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", ...
            coordinate_transformation_mode, 'coordinate_transformation_mode', 'half_pixel'))];
        coordinate_transformation_mode = "half_pixel";
    end
    % if the nearest_mode is "floor", "ceil", "round_prefer_floor" , set the nearest_mode to be "onnx-10", 
    % and thrown out a warning
    if ismember(nearest_mode, ["floor", "ceil", "round_prefer_floor"])
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", nearest_mode, 'nearest_mode', 'onnx-10'))];
    end
    % if the mode is "cubic", set the mode to be "linear", and throw out a warning
    if isequal(mode, "cubic")
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", 'cubic', 'mode', 'linear'))];
        mode = "linear";
    end
    % if the antialias is true, set it to be false, and thrown out a warning
    if antialias
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport",'1', 'antialias','0'))];
    end
    if ~isempty(axes)
        nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedAttribute",'axes'));
        return;
    end
    % if the keep_aspect_ratio_policy is "not_larger", "not_smaller", "keeping", 
    % set it to be "stretch", and thrown out a warning
    if ismember(keep_aspect_ratio_policy, ["not_larger", "not_smaller", "keeping"])
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", ...
            keep_aspect_ratio_policy, 'keep_aspect_ratio_policy', 'stretch'))];
        keep_aspect_ratio_policy = "stretch";
    end
    
    % Gen code
    Y = node.output{1};
    X = node.input{1};
    roi = node.input{2};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    if isempty(roi)
        VarRoi = "dlarray([])";
    else
        [VarRoi, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, roi);        
    end
    % scales can be the empty string
    if isempty(node.input{3})
        scalesStr = "dlarray([])";
    else
        scalesStr  = node.input{3};
        [scalesStr, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, scalesStr);

    end
    % sizes can be the empty string or missing
    if numel(node.input)<4 || isempty(node.input{4})
        sizesStr = "dlarray([])";
    else
        sizesStr  = node.input{4};
        [sizesStr, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, sizesStr);

    end
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, %s] = %sprepareResize11Args(%s, %s, %s, "%s", "%s", "%s", %s);\n', ...
        NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarRoi, scalesStr, sizesStr, coordinate_transformation_mode, mode, nearest_mode, NumDimX),...
        sprintf('if isempty(DLTScales)\n'),...
        sprintf('%s = dlresize(%s, ''OutputSize'', DLTSizes, ''DataFormat'', dataFormat, ''Method'', Method, ''GeometricTransformMode'', GeometricTransformMode, ''NearestRoundingMode'', NearestRoundingMode);\n', ...
        VarY, VarX),...
        sprintf('else\n'),...
        sprintf('%s = dlresize(%s, ''Scale'', DLTScales, ''DataFormat'', dataFormat, ''Method'', Method, ''GeometricTransformMode'', GeometricTransformMode, ''NearestRoundingMode'', NearestRoundingMode);\n', ...
        VarY, VarX),...
        sprintf('end\n'),...
        ];
    nodeTranslation.IncludedFunctionNames = "prepareResize11Args";
    nodeTranslation.MCode = string(command);
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end

function issues = iValidateInput(node, mode, cubic_coeff_a, coordinate_transformation_mode, exclude_outside, extrapolation_value, nearest_mode)
checks = {
    mode,                           ["nearest", "linear", "cubic"],                              "mode"
    cubic_coeff_a,                  -0.75,                                                       "cubic_coeff_a"
    coordinate_transformation_mode, ["pytorch_half_pixel", "half_pixel", "asymmetric"],          "coordinate_transformation_mode"
    exclude_outside,                0,                                                           "exclude_outside"
    extrapolation_value,            0.0,                                                         "extrapolation_value"
    nearest_mode,                   ["floor", "ceil", "round_prefer_floor", "round_prefer_ceil"], "nearest_mode"
};
issues = [];
for i = 1:size(checks, 1)
    if ~ismember(checks{i,1}, checks{i,2})
        issues = [issues
            nnet.internal.cnn.onnx.NodeTranslationError(node,...
                message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", string(checks{i,1}), checks{i,3}))]; %#ok<AGROW>
    end
end
end
