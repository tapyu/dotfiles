classdef TranslatorForResize < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021-2026 The MathWorks, Inc.    

    properties
        % Operator Attributes
        antialias
        axes
        coordinate_transformation_mode
        cubic_coeff_a
        exclude_outside
        extrapolation_value
        keep_aspect_ratio_policy
        mode
        nearest_mode

        % Cached layer and issues
        Layer
        Issues
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            if this.GraphProtoManager.OpsetVersion == 10
                AttributeTable = cell2table({
                    "mode"	       "STRING"   true    "nearest"
                    });
                this.mode = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
                this.coordinate_transformation_mode = '';
                this.nearest_mode = '';
                this.keep_aspect_ratio_policy = "stretch";
                this.axes = [];
                this.antialias = 0;
            elseif ismember(this.GraphProtoManager.OpsetVersion, 11:17)
                % Attributes for opset version 11-17.
                AttributeTable = cell2table({
                    "coordinate_transformation_mode"	"STRING"      true    "half-pixel"
                    "cubic_coeff_a"                     "FLOAT"       true    -0.75
                    "exclude_outside"                   "INT"         true    0
                    "extrapolation_value"               "FLOAT"       true    0
                    "mode"                              "STRING"      true    "nearest"
                    "nearest_mode"                      "STRING"      true    "round_prefer_floor"
                    });
                [this.coordinate_transformation_mode ,~,~,~, this.mode, this.nearest_mode]  = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
                this.keep_aspect_ratio_policy = "stretch";
                this.axes = [];
                this.antialias = 0;
            else
                % Attributes for opset version 18+.
                AttributeTable = cell2table({
                    "antialias"                         "INT"         true    0
                    "axes"                              "INTS"        true    []
                    "coordinate_transformation_mode"	"STRING"      true    "half-pixel"
                    "cubic_coeff_a"                     "FLOAT"       true    -0.75
                    "exclude_outside"                   "INT"         true    0
                    "extrapolation_value"               "FLOAT"       true    0
                    "keep_aspect_ratio_policy"          "STRING"      true    "stretch"
                    "mode"                              "STRING"      true    "nearest"
                    "nearest_mode"                      "STRING"      true    "round_prefer_floor"
                    });
                [this.antialias, this.axes, this.coordinate_transformation_mode ,~,~,~, this.keep_aspect_ratio_policy, this.mode, this.nearest_mode]  = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            end
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            [inputTensorFormats, outputTensorFormats] = propagateSISOPassthroughOp(this, direction, inputTensorFormats, outputTensorFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues. 
        
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            % Resize2dlayer and resize3dlayer accept unformatted inputs. 
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if ~nnet.internal.cnn.onnx.isInstalledIPT()
                issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message('nnet_cnn_onnx:onnx:noIPTForOperator', 'Resize'));
            elseif ismember(inputTensorFormats(1), ["BCSS", "BCSSS"]) ...
                    && outputTensorFormats(1)~="" ...
                    && all((~cellfun(@isempty, this.Node.input(2:numel(inputTensorFormats)))) == isNodeInputInitializer(this.GraphProtoManager, this.Node, 2:numel(inputTensorFormats)))
                % Only translate resize into resize layers when scales or sizes is a initializer
                % Thus check if the last input is an initializer
                LayerName = this.Node.name;
                % Validate and return the attributes supported by MATLAB.
                [this.coordinate_transformation_mode, coordTransModeIssue] = iValidateAndReturnCoordinateTransformMode(this.coordinate_transformation_mode, this.Node);
                [this.nearest_mode, nearestModeIssue]                      = iValidateAndReturnNearestMode(this.nearest_mode, this.Node);
                [this.mode, modeIssue]                                     = iValidateAndReturnMode(this.mode, this.Node);
                [this.keep_aspect_ratio_policy, keepAspectRatioIssue]      = iValidateAndReturnKeepAspectRatio(this.keep_aspect_ratio_policy, this.Node);
                [antialiasIssue]                                           = iValidateAntialias(this.antialias, this.Node);
                % If any issues require a placeholder layer, return an empty layer and
                % the list of all issues
                issues = [coordTransModeIssue nearestModeIssue modeIssue keepAspectRatioIssue antialiasIssue];
                placeholders = arrayfun(@(x) isa(x, "nnet.internal.cnn.onnx.NodeTranslationError"), issues);
                if any(placeholders)
                    Layer = [];
                    return;
                else
                    % For opset 11+
                    % The maximum number of inputs is 4. Either scales or
                    % sizes has to be specified. 
                    % [x, empty ROI, empty scales, sizes] or [x, empty ROI, scales]
                    % For opset 10
                    % [x, scale]
                    if numel(this.Node.input) == 2
                        scaleName = this.Node.input{2};
                        scales = single(initializerRawData(this.GraphProtoManager, scaleName));
                        [Layer, issue] = iCreateLayerFromONNX(this, this.coordinate_transformation_mode, this.nearest_mode, this.mode, this.axes, 'Scale', scales, LayerName, this.Node);
                    elseif numel(this.Node.input) == 3
                        scaleName = this.Node.input{3};
                        scales = single(initializerRawData(this.GraphProtoManager, scaleName));
                        [Layer, issue] = iCreateLayerFromONNX(this, this.coordinate_transformation_mode, this.nearest_mode, this.mode, this.axes, 'Scale', scales, LayerName, this.Node);
                    else
                        outputSizeName = this.Node.input{4};
                        outputSize = single(initializerRawData(this.GraphProtoManager, outputSizeName));
                        [Layer, issue] = iCreateLayerFromONNX(this, this.coordinate_transformation_mode, this.nearest_mode, this.mode, this.axes, 'OutputSize', outputSize, LayerName, this.Node);
                    end
                    if ~isempty(issue)
                        issues = [issues issue];
                        Layer = [];
                    end
                end
            end
        end
    end
end

function [layer, issue] = iCreateLayerFromONNX(this, coordinateTransformMode, nearestMode, mode, axes, type, value, LayerName, node)
% iCreateLayerFromONNX returns a layer. Inputs arguments are coordinate
% transform mode, nearest rounding mode, interpolation method,
% type (Scale or OutputSize), value of scale or output size, and layer
% name.
layer = [];
[value, issue] = iGetValue(axes, type, value, node);
if ~isempty(issue)
    return;
end
if numel(value) == 2
    mode = iConvertMethodFromONNX(mode, false);
    [layer, issue] = constructLayer(this, 'resize2dLayer', LayerName, node, 'Name', ...
        LayerName, type, value, 'Method', mode, 'NearestRoundingMode', nearestMode, ...
        'GeometricTransformMode', coordinateTransformMode);
elseif numel(value) == 3
    mode = iConvertMethodFromONNX(mode, true);
    [layer, issue] = constructLayer(this, 'resize3dLayer', LayerName, node, 'Name', ...
        LayerName, type, value, 'Method', mode, 'NearestRoundingMode', nearestMode, ...
        'GeometricTransformMode', coordinateTransformMode);
else
    issue = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedNumberOfElementsScale"));
end
end

function [value,issue] = iGetValue(axes, type, value, node)
issue = [];
supportedAxes = {[2,3,4], [-3,-2,-1], [2,3], [-2,-1]};
if isempty(axes)
    % if axes is not specified, uses the rank of input to compute value. 
    if ~(isequal(value(1),1) && isequal(value(2),1)) && isequal(type, "Scale")
        % B and C dimension could not be scaled
        issue = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedNumberOfElementsScale"));
        return;
    end
    if numel(value) == 4 
        value = [value(end-1) value(end)];
    elseif numel(value) == 5
        value = [value(end-2) value(end-1) value(end)];
    else
        issue = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedNumberOfElementsScale"));
        return;
    end
else
    % if axes is specified, uses the axes instead of rank of input. 
    % only translates to resize2dlayer or resize3dlayer when 2 or 3
    % dimensions are scaled
    % the number of value and axes should be the same
    if all(cellfun(@(x) isequal(axes, x), supportedAxes) == 0) || ~isequal(numel(value), numel(axes))
        issue = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedNumberOfElementsScale"));
    end
end
end

function mode = iConvertMethodFromONNX(mode, has3dInput)
% Converts interpolation method string to layer type.
if strcmpi(mode,'linear')
    if has3dInput
        mode = 'trilinear';
    else
        mode = 'bilinear';
    end
end
end

function [coordinateTransformMode, issue] = iValidateAndReturnCoordinateTransformMode(coordinateTransformMode, node)
coordinateTransformMode = char(coordinateTransformMode);
issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
switch coordinateTransformMode
    case 'asymmetric'
        coordinateTransformMode = 'asymmetric';
    case 'half_pixel'
        coordinateTransformMode = 'half-pixel';
    case 'pytorch_half_pixel'
        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport",...
            string(coordinateTransformMode),"nearest_mode","half-pixel"));
        coordinateTransformMode = 'half-pixel';
    case ''
        coordinateTransformMode = 'half-pixel';
    otherwise
        issue = nnet.internal.cnn.onnx.NodeTranslationError(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue",...
            string(coordinateTransformMode),"coordinate_transformation_mode", node.op_type));
        
end
end

function [issue] = iValidateAntialias(antialias, node)
issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
if antialias ~= 0
issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
    message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", '1', 'antialias', '0'));
end
end
function [mode, issue] = iValidateAndReturnMode(mode, node)
issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
supportedModes = ["nearest", "linear", "cubic"];
if ~ismember(string(mode), supportedModes)
    issue = nnet.internal.cnn.onnx.NodeTranslationError(node, ...
        message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", string(mode), "mode", node.op_type));
end
if isequal(string(mode),"cubic")
    mode = 'linear';
    issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", "cubic", "mode", "linear"));
end
end

function [aspect_ratio, issue]= iValidateAndReturnKeepAspectRatio(aspect_ratio, node)
issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
supportedAspectRatio = "stretch";
if ~isequal(string(aspect_ratio), supportedAspectRatio)
    issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
        message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", string(aspect_ratio), "keep_asepct_ratio_policy", "stretch"));
    aspect_ratio = "stretch";
end
end

function [nearestMode, issue] = iValidateAndReturnNearestMode(nearestMode, node)
nearestMode = char(nearestMode);
issue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
switch nearestMode
    case 'round_prefer_ceil'
        nearestMode = 'round';
    case {'','round_prefer_floor'}
        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport",...
            string(nearestMode),"nearest_mode","onnx-10"));
        nearestMode = 'onnx-10';
    case 'floor'
        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport",...
            string(nearestMode),"nearest_mode","onnx-10"));
        nearestMode = 'onnx-10';
    case 'ceil'
        issue = nnet.internal.cnn.onnx.NodeTranslationWarning(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport",...
            string(nearestMode),"nearest_mode","onnx-10"));
        nearestMode = 'onnx-10';
    otherwise
        issue = nnet.internal.cnn.onnx.NodeTranslationError(node, ...
            message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue",...
            string(nearestMode),"nearest_mode", node.op_type));
end
end
