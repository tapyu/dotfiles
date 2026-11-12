classdef TranslatorForConv < nnet.internal.cnn.onnx.OperatorTranslator
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties
        % Operator Attributes
        auto_pad
        dilations
        group
        kernel_shape
        pads
        strides
        
        % Cached layer and issues
        Layer
        Issues
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
                "pads"          "INTS"      true    []
                "strides"       "INTS"      true    []
                });
            % Parse the attributes
            [this.auto_pad, this.dilations, this.group, this.kernel_shape, this.pads, this.strides] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
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
                weightDim = initializerSize(this.GraphProtoManager, this.Node.input{2}); % NumFilters x NumChannels/NumGroups x H x W (x D, if 3d conv)
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
            % To be fully supported, in1 and out must have labels, and
            % inputs 2 and 3(if present) must be initializers:
            isIOLabeled = inputTensorFormats(1)~="" && outputTensorFormats(1)~="";
            isWInitializer = isNodeInputInitializer(this.GraphProtoManager, this.Node, 2);
            isIn3EmptyOrInitializer = numel(this.Node.input)<3 || isempty(this.Node.input{3}) || isNodeInputInitializer(this.GraphProtoManager, this.Node, 3);
            supported = isIOLabeled && isWInitializer && isIn3EmptyOrInitializer;
            if supported
                LayerName = this.Node.name;

                % To get number of filters
                weight_name = this.Node.input{2};
                weightDim = initializerSize(this.GraphProtoManager, weight_name); %NumFilters x NumChannels/NumGroups x H x W (x D, if 3d conv)
                
                % Get the filter size
                if isempty(this.kernel_shape)
                    this.kernel_shape = weightDim(3:end);
                end
                FilterSize = this.kernel_shape; %h or [h w] or [h w d]
                numSpatialDims = numel(FilterSize);
                
                % Create a vector of ones for spatial dims
                if numSpatialDims == 3
                    spatialDimOnes = [1 1 1]; % [H W D]
                elseif numSpatialDims == 2
                    spatialDimOnes = [1 1];   % [H W]
                else
                    spatialDimOnes = 1;
                end
                
                % Get the number of Weight groups
                numWtGroups = this.group;
                
                % Get the stride along each spatial dimension
                if isempty(this.strides)
                    this.strides = spatialDimOnes; % 1d: [1], 2d: [1 1], 3d: [1 1 1]
                end
                
                % Handle padding
                if ~isempty(this.pads)
                    Padding = this.pads;
                    Padding = iConvertONNXToMATLABPadding(Padding, numSpatialDims);
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
                                message('nnet_cnn_onnx:onnx:AutoPadSameLower', LayerName))];
                        case 'VALID'
                            Padding = getDefaultPadding(numSpatialDims);
                        case 'NOTSET'
                            % this.pads is not explicitly set at this point so default is used
                            Padding = getDefaultPadding(numSpatialDims);
                        otherwise
                            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                                message('nnet_cnn_onnx:onnx:UnknownAttributeSetting', 'auto_pad', 'Conv'))];
                    end
                end
                
                % Get this.dilations
                if isempty(this.dilations)
                    this.dilations = spatialDimOnes; % 1d: [1], 2d: [1 1], 3d: [1 1 1]
                end
                
                if numel(weightDim) > 5
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:ConvDim'))];
                    Layer = [];
                    return;
                end
                NumFilters = double(weightDim(1));
                
                % Import weights
                %assume the 2nd input is W and the 3rd input is bias.
                if numel(this.Node.input) > 2
                    biasName = this.Node.input{3};
                    bias = single(initializerRawData(this.GraphProtoManager, biasName));
                    bias = reshape(bias, [spatialDimOnes NumFilters]);
                else
                    bias = zeros([spatialDimOnes NumFilters], 'single');
                end
                
                rawWeights = single(initializerRawData(this.GraphProtoManager, weight_name));     % Raw data is row-major
                if numSpatialDims == 3
                    weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [D,W,H,C,F]
                    weights = permute(weights, [3,2,1,4,5]);                    % DWHCF -> HWDCF
                elseif numSpatialDims ==2
                    weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [W,H,C,F]
                    weights = permute(weights, [2,1,3,4]);                      % WHCF -> HWCF
                else
                    weights = reshape(rawWeights, fliplr(weightDim));           % Reshape raw weights to [H, C, F]
                end
                
                if numWtGroups == 1
                    % Create ordinary convolution
                    % Construct cell array of input arguments to the conv2d or conv3d layer
                    args = {FilterSize, NumFilters, 'Stride', this.strides, 'DilationFactor', this.dilations,...
                        'Padding', Padding, 'Weights', weights, 'Bias', bias, 'Name', LayerName};
                    if numSpatialDims == 3
                        [Layer, constructionIssue] = constructLayer(this, 'convolution3dLayer', LayerName, this.Node, args{:});
                    elseif numSpatialDims == 2
                        [Layer, constructionIssue] = constructLayer(this, 'convolution2dLayer', LayerName, this.Node, args{:});
                    else
                        [Layer, constructionIssue] = constructLayer(this, 'convolution1dLayer', LayerName, this.Node, args{:});
                    end
                    issues = [issues constructionIssue];
                else
                    % 3d grouped convolution is not supported
                    if numSpatialDims == 3 || numSpatialDims == 1
                        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node, ...
                            message('nnet_cnn_onnx:onnx:GroupedConvolution3D'))];
                        Layer = [];
                        return;
                    end
                    
                    % Create 2d grouped convolution
                    NumFiltersPerGroup = NumFilters/numWtGroups;
                    NumChannelsPerGroup = double(weightDim(2));
                    weights = reshape(weights, [FilterSize,NumChannelsPerGroup,...
                        NumFiltersPerGroup,numWtGroups]);
                    bias = reshape(bias, [spatialDimOnes,NumFiltersPerGroup,numWtGroups]);
                    
                    [Layer, constructionIssue] = constructLayer(this, 'groupedConvolution2dLayer', LayerName, this.Node, FilterSize, NumFiltersPerGroup, numWtGroups,...
                        'Stride', this.strides,...
                        'DilationFactor', this.dilations, ...
                        'Padding', Padding,...
                        'Weights', weights, ...
                        'Bias', bias, ...
                        'Name', LayerName);
                    issues = [issues constructionIssue];
                end
            end
        end
    end
end

function Padding = iConvertONNXToMATLABPadding(Padding, numSpatialDims)
if numSpatialDims == 3
    % ONNX:   [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r k]
    % MATLAB: [t l f; b r k]
    Padding = iFixTwoElementPadding(Padding, numSpatialDims);
    Padding = Padding([1 3 5; 2 4 6]);
elseif numSpatialDims == 2
    % ONNX:   [H_b,W_b,H_end,W_end] ==> [t l b r]
    % MATLAB: [t b l r]
    Padding = iFixTwoElementPadding(Padding, numSpatialDims);
    Padding = Padding([1,3,2,4]);
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

function Padding = iFixTwoElementPadding(Padding, numSpatialDims)
% If Padding has only two elements, interpret them as the start and end
% padding to use along every spatial dimension. Return a full 4- or
% 6- element vector with the duplicated values.
if numel(Padding) == 2
    startPad = Padding(1);
    endPad = Padding(2);
    if numSpatialDims == 3
        % [start end] -> [start start start; end end end]
        Padding = [repelem(startPad, 3); repelem(endPad, 3)];
    elseif numSpatialDims == 2
        % [start end] -> [start start; back back]
        Padding = [repelem(startPad, 2) repelem(endPad, 2)];
    end
end

end
