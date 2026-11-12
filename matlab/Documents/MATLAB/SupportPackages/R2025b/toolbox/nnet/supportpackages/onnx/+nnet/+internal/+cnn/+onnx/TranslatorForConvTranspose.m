classdef TranslatorForConvTranspose < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX ConvTranspose operators into MATLAB layers
    
    % Copyright 2021-2026 The MathWorks, Inc.
    
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
                "dilations"     "INTS"      true    1
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
                    if numel(weightDim)==4
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
                % Determine whether the input is 2d or 3d
                weight_name = this.Node.input{2};
                weightDim   = initializerSize(this.GraphProtoManager, weight_name); % CFHW (2d) or CFHWD (3d)
                has3dImageInput = (numel(weightDim) == 5);
                
                % get FilterSize from weight tensor shape
                NumFilters 	= double(weightDim(2));
                Height     	= double(weightDim(3));
                Width     	= double(weightDim(4));
                
                % Create a vector of ones for spatial dims
                if has3dImageInput % "BCSSS"
                    spatialDimOnes = [1 1 1]; % [H W D]
                    weightPerm = [3 2 1 4 5]; % To turn DWHFC -> HWDFC col-major
                else % "BCSS"
                    spatialDimOnes = [1 1];   % [H W]
                    weightPerm = [2 1 3 4];   % To turn WHFC -> HWFC col-major
                end
                
                % dilations
                if  ~all(this.dilations == 1)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node, ...
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
                            Padding = getDefaultPadding(has3dImageInput);
                        case 'NOTSET'
                            % Pads is not explicitly set at this point so default is used
                            Padding = getDefaultPadding(has3dImageInput);
                        otherwise
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:UnknownAttributeSetting', 'auto_pad'))];
                    end
                end
                
                % Convert ONNX padding to DLT cropping
                [Cropping, cropIssues] = getCroppingFromPadding(this.Node, Padding, has3dImageInput);
                if ~isempty(cropIssues)
                    issues = [issues cropIssues];
                    Layer = [];
                    return;
                end
                
                % strides
                if isempty(this.strides)
                    Strides = spatialDimOnes; % 2d: [1 1], 3d: [1 1 1]
                else
                    Strides = this.strides;
                end
                
                % output_padding and output_shape
                if ~isempty(this.output_padding)  && ~all(this.output_padding == 0) % output_padding of all zeros should not have any effect
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'output_padding'))];
                    Layer = [];
                    return;
                end
                if ~isempty(this.output_shape)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'output_shape'))];
                    Layer = [];
                    return;
                end
                
                % Get bias if it's an initializer
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
                
                % Get weights
                % ONNX:     CFHW row-major
                % MATLAB:   HWFC col-major
                Weights = single(initializerRawData(this.GraphProtoManager, weight_name));     % CFHW (2d) or CFHWD (3d) row-major
                Weights = reshape(Weights, fliplr(weightDim));                  % WHFC (2d) or DWHFC (3d) col-major
                Weights = permute(Weights, weightPerm);                         % HWFC (2d) or HWDFC (3d) col-major
                
                % Create the layer
                if has3dImageInput
                    Depth = double(weightDim(5));
                    FilterSize = [Height, Width, Depth];
                    % Create the DLT layer
                    [Layer, constructionIssue] = constructLayer(this, 'transposedConv3dLayer', this.LayerName, this.Node, FilterSize, NumFilters,...
                        'Weights', Weights, 'Bias', Bias, 'Stride', Strides, 'Cropping', Cropping, 'Name', this.LayerName);
                else
                    FilterSize 	= [Height, Width];
                    % Create the DLT layer
                    [Layer, constructionIssue] = constructLayer(this, 'transposedConv2dLayer', this.LayerName, this.Node, FilterSize, NumFilters,...
                        'Weights', Weights, 'Bias', Bias, 'Stride', Strides, 'Cropping', Cropping, 'Name', this.LayerName);
                end
                issues = [issues constructionIssue];
                
                if isempty(Layer)
                    Layer = [];
                    return;
                end
            end
        end
    end
    
    methods(Access=protected)
        function tf = isLabeled(~, inputTensorFormat, outputTensorFormat)
            tf = ~isequal(inputTensorFormat, "") &&  ~isequal(outputTensorFormat, "");
        end
        
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            tf = false;
            supportedFormats = ["BCSS", "BCSSS"];
            if ismember(inputTensorFormat, supportedFormats) && strcmp(inputTensorFormat, outputTensorFormat)
                tf = true;
            end
        end
    end
end

function Padding = iConvertONNXToMATLABPadding(Padding, has3dImageInput)
if has3dImageInput
    % ONNX:   [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r k]
    % MATLAB: [t l f; b r k]
    Padding = iFixTwoElementPadding(Padding, has3dImageInput);
    Padding = Padding([1 3 5; 2 4 6]);
else
    % ONNX:   [H_b,W_b,H_end,W_end] ==> [t l b r]
    % MATLAB: [t b l r]
    Padding = iFixTwoElementPadding(Padding, has3dImageInput);
    Padding = Padding([1,3,2,4]);
end
end

function Padding = getDefaultPadding(has3dImageInput)
if has3dImageInput
    Padding = [0 0 0; 0 0 0]; % [t l f; b r k]
else
    Padding = [0 0 0 0]; % [t b l r]
end
end

function [Cropping, issues] = getCroppingFromPadding(node, Padding, NetworkHas3dImageInput)
% Convert ONNX padding to DLT cropping:
% ONNX   2d : [H_b,W_b,H_end,W_end] ==> [t l b r]
%        3d : [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r k]
% MATLAB 2d : [t=b l=r]
%        3d : [t=b l=r f=b]

issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
if NetworkHas3dImageInput
    Cropping = Padding([1,4,2,5,3,6]);
    if Cropping(1)~=Cropping(2) || Cropping(3)~=Cropping(4) || Cropping(5) ~= Cropping(6)
        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(node, ...
            message('nnet_cnn_onnx:onnx:ConvTransposeCropping'))];
    end
    Cropping = Cropping([1 3 5]);
    
else
    Cropping = Padding([1,3,2,4]);
    if Cropping(1)~=Cropping(2) || Cropping(3)~=Cropping(4)
        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(node,...
            message('nnet_cnn_onnx:onnx:ConvTransposeCropping'))];
    end
    Cropping = Cropping([1 3]);
    
end

end
