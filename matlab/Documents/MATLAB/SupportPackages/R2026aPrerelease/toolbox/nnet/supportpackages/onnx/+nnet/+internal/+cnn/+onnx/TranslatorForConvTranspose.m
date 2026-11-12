classdef TranslatorForConvTranspose < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX ConvTranspose operators into MATLAB layers
    
    % Copyright 2021-2025 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        auto_pad
        dilations
        group
        kernel_shape
        output_padding
        output_shape
        pads
        strides
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.IsSeedOperator = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "auto_pad"      "STRING"    true    "NOTSET"
                "dilations"     "INTS"      true    []
                "group"         "INT"       true    1
                "kernel_shape"  "INTS"      true    []
                "output_padding" "INTS"     true    []
                "output_shape"  "INTS"      true    []
                "pads"          "INTS"      true    []
                "strides"       "INTS"      true    []
                });
            % Parse the attributes
            [this.auto_pad, this.dilations, this.group, this.kernel_shape, this.output_padding, this.output_shape, this.pads, this.strides] = ...
                nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
            % Set the layer name
            this.LayerName = this.Node.name;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputTensorFormats, outputTensorFormats] = propagateTensorFormats(this, direction, inputTensorFormats, outputTensorFormats)
            % Try to derive formats from the dimensionality of W
                if isNodeInputInitializer(this.GraphProtoManager, this.Node, 2)
                    % W is an initializer
                    weightDim = initializerSize(this.GraphProtoManager, this.Node.input{2});     % C x NumFeatureMaps/NumGroups x H x W (x D, if 3d conv)
                    if numel(weightDim)==3
                        inputTensorFormats(1) = "BCT";
                        outputTensorFormats(1) = "BCT";
                    elseif numel(weightDim)==4
                        inputTensorFormats(1) = "BCSS";
                        outputTensorFormats(1) = "BCSS";
                    elseif numel(weightDim)==5
                        inputTensorFormats(1) = "BCSSS";
                        outputTensorFormats(1) = "BCSSS";
                    end
                end
            end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % Inputs 2 and 3 (if present) must be initializers, and the
            % first I/O must have supported formats.
            supported = isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) ...
                && (numel(this.Node.input)==2 || isempty(this.Node.input{3}) || isNodeInputInitializer(this.GraphProtoManager, this.Node, 3)) ...
                && supportedFormat(this, inputTensorFormats(1), outputTensorFormats(1));
            if supported
                % Determine whether the input is 1d or 2d or 3d
                weight_name = this.Node.input{2};
                weightDim   = initializerSize(this.GraphProtoManager, weight_name); % CFT (1d) or CFHW (2d) or CFHWD (3d)
                if numel(weightDim) < 3 || numel(weightDim) > 5
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:ConvTransposeDim'))];                          
                    Layer = [];
                    return;
                end
                
                % Get the filter size
                if isempty(this.kernel_shape)
                    this.kernel_shape = weightDim(3:end);
                end
                FilterSize = this.kernel_shape; % h or [h w] or [h w d]                
                numSpatialDims = numel(FilterSize);
                NumFilters 	= double(weightDim(2));
                
                % Create a vector of ones for spatial dims
                if numSpatialDims == 3
                    spatialDimOnes = [1 1 1]; % [H W D]
                elseif numSpatialDims == 2
                    spatialDimOnes = [1 1];   % [H W]
                else
                    spatialDimOnes = 1;       % [H]
                end
                
                % Strides
                if isempty(this.strides)
                    Strides = spatialDimOnes; % 1d: [1], 2d: [1 1], 3d: [1 1 1]
                else
                    Strides = this.strides;
                end
                
                % Handle padding
                if ~isempty(this.pads)
                    Padding = this.pads;
                    if this.auto_pad~="NOTSET"
                        issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                            message('nnet_cnn_onnx:onnx:AutoPadAndPadDefined'))];
                    end
                else
                    switch this.auto_pad
                        case 'SAME_UPPER'
                            Padding = 'same';
                        case 'SAME_LOWER'
                            Padding = 'same';
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:AutoPadSameLower', this.LayerName))];
                        case 'VALID'
                            Padding = getDefaultPadding(numSpatialDims);
                        case 'NOTSET'
                            % Pads is not explicitly set at this point so default is used
                            Padding = getDefaultPadding(numSpatialDims);
                        otherwise
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:UnknownAttributeSetting', 'auto_pad', 'ConvTranspose'))];
                    end
                end
                
                % Get this.dilations
                if isempty(this.dilations)
                    this.dilations = spatialDimOnes; % 1d: [1], 2d: [1 1], 3d: [1 1 1]
                end
                
                % dilations
                if  ~all(this.dilations == 1)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedDilation'))];
                    Layer = [];
                    return;
                end
                
                % group
                if this.group > 1
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:Max1WeightGroup'))];
                    Layer = [];
                    return;
                end
                
                % Import weights
                % assume the 2nd input is W and the 3rd input is bias.
                % Get bias if it's an Initializer
                if numel(this.Node.input) > 2
                    biasName = this.Node.input{3};
                    if isempty(initializerSize(this.GraphProtoManager, biasName))
                        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                            message('nnet_cnn_onnx:onnx:ConvTransposeBias'))];
                        Layer = [];
                        return;
                    end
                    Bias = single(initializerRawData(this.GraphProtoManager, biasName));
                    Bias = reshape(Bias, [spatialDimOnes, NumFilters]);
                else
                    Bias = zeros([spatialDimOnes, NumFilters], 'single');
                end
                
                rawWeights = single(initializerRawData(this.GraphProtoManager, weight_name));     % Raw data is row-major
                if numSpatialDims == 3
                    Weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [D,W,H,C,F]
                    Weights = permute(Weights, [3,2,1,4,5]);                    % DWHCF -> HWDCF
                elseif numSpatialDims ==2
                    Weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [W,H,C,F]
                    Weights = permute(Weights, [2,1,3,4]);                      % WHCF -> HWCF
                else
                    Weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [H, C, F]
                end

                % Convert ONNX padding to DLT cropping
                Cropping = getCroppingFromPadding(Padding, numSpatialDims);                
                % output_padding and output_shape
                if ~isempty(this.output_padding) && ~all(this.output_padding == 0) % output_padding of all zeros should not have any effect
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'output_padding', this.Node.op_type))];
                    Layer = [];
                    return;
                end
                if ~isempty(this.output_shape)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'output_shape', this.Node.op_type))];
                    Layer = [];
                    return;
                end
                
                % Create the layer
                switch numSpatialDims
                    case 3
                        % Create the transposedConv3dLayer DLT layer
                    [Layer, constructionIssue] = constructLayer(this, 'transposedConv3dLayer', this.LayerName, this.Node, FilterSize, NumFilters,...
                        'Weights', Weights, 'Bias', Bias, 'Stride', Strides, 'Cropping', Cropping, 'Name', this.LayerName);
                    case 2                 
                        % Create the transposedConv2dLayer DLT layer
                    [Layer, constructionIssue] = constructLayer(this, 'transposedConv2dLayer', this.LayerName, this.Node, FilterSize, NumFilters,...
                        'Weights', Weights, 'Bias', Bias, 'Stride', Strides, 'Cropping', Cropping, 'Name', this.LayerName);
                    case 1
                        % % Create the transposedConv1dLayer DLT layer
                        [Layer, constructionIssue] = constructLayer(this, 'transposedConv1dLayer', this.LayerName, this.Node, FilterSize, NumFilters,...
                            'Weights', Weights, 'Bias', Bias, 'Stride', Strides, 'Cropping', Cropping, 'Name', this.LayerName);
                end
                issues = [issues constructionIssue];
            end
        end
    end
    
    methods(Access=protected)
        function tf = isLabeled(~, inputTensorFormat, outputTensorFormat)
            tf = ~isequal(inputTensorFormat, "") &&  ~isequal(outputTensorFormat, "");
        end
        
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCT", "BCSS", "BCSSS"];
            if ismember(inputTensorFormat, supportedFormats) && strcmp(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end
    end
end

function Padding = getDefaultPadding(numSpatialDims)
if numSpatialDims == 3
    Padding = [0 0 0; 0 0 0]; % [t l f; b r k]
elseif numSpatialDims == 2
    Padding = [0 0 0 0]; % [t b l r]
else
    Padding = [0 0]; %[t b]
end
end

function Cropping = getCroppingFromPadding(Padding, numSpatialDims)
% Convert ONNX padding to DLT cropping:
% ONNX   1d : [H_b,H_end] ==> [l r] 
%        2d : [H_b,W_b,H_end,W_end] ==> [t l b r]
%        3d : [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r bk]
% MATLAB 1d : [l r] 
%        2d : [t b l r]
%        3d : [t l f; b r bk]

if ~isnumeric(Padding)
    % Padding = 'same'
    Cropping = Padding;
else
    switch numSpatialDims
        case 1
            Cropping = Padding([1,2]);
        case 2
    Cropping = Padding([1,3,2,4]);
        case 3
            Cropping = Padding([1,2,3;4,5,6]);            
    end
end
end
