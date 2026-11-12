classdef TranslatorForLayerNormalization < nnet.internal.cnn.onnx.OperatorTranslator
    %TranslatorForLayerNormalization A class to translate ONNX
    % LayerNormalization operator into MATLAB layer

    %   Copyright 2024 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator attributes
        axis
        epsilon
        stash_type

        % Other properties
        LayerName
    end

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "axis"         "INT"        true    -1
                "epsilon"      "FLOAT"      true    1e-5
                "stash_type"   "INT"        true    1
                });
            % Parse the attributes
            [this.axis, this.epsilon, this.stash_type] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);

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
            [inputTensorFormats, outputTensorFormats] = propagateSISOPassthroughOp(this, direction, inputTensorFormats, outputTensorFormats);
        end

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % The MATLAB LayerNormLayer only support single output. The
            % optional ouputs in ONNX LayerNormlization are not supported in MATLAB
            if isscalar(this.Node.output) || ~any(outputUsedInNetwork(this, [2,3]))
                % Validate the input/output formats
                if ~supportedFormat(this, inputTensorFormats, outputTensorFormats)
                    Layer = [];
                    issues = nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedInputTensorFormat', inputTensorFormats, 'LayerNormalization'));
                    return;
                end
                
                OpDimension = getOperationDimension(this,inputTensorFormats);
                if isempty(OpDimension)
                    Layer = [];
                    return;
                end

                % input 2 and optional input 3 must be initializers.
                hasBias = numel(this.Node.input) == 3;
                if ~inputsInitializers(this, hasBias)
                    Layer = [];
                    if hasBias
                        inputListStr = join(["'scale'", "'B'"], ',');
                    else
                        inputListStr = "'scale'";
                    end
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:InputFromInitializer', inputListStr, 'LayerNormalization'));
                    return;
                end

                % Casting inputs to different datatype in the layer "stash_type" is not supported
                if this.stash_type
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node, ...
                            message('nnet_cnn_onnx:onnx:IgnoredAttributes',"stash_type",this.Node.op_type))];
                end
                % Epsilon must be a postive number
                if this.epsilon <= 0
                    Epsilon = 1e-5;
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                        message('nnet_cnn_onnx:onnx:BadEpsilon'))];
                else
                    Epsilon = double(this.epsilon);
                end
                ScaleName       = this.Node.input{2};
                if hasBias
                    BiasName    = this.Node.input{3};
                end
                
                NumChannels     = double(initializerSize(this.GraphProtoManager, ScaleName));
                Offset          = [];
                if hasBias
                    Offset      = reshapeData(this, single(initializerRawData(this.GraphProtoManager, BiasName)),  ...
                    NumChannels, inputTensorFormats(1));
                end
                Scale       = reshapeData(this, single(initializerRawData(this.GraphProtoManager, ScaleName)), ...
                    NumChannels, inputTensorFormats(1));

                [Layer, constructionIssue] = constructLayer(this, 'layerNormalizationLayer', ...
                    this.LayerName, this.Node, 'Name', this.LayerName, 'Offset', Offset, ...
                    'Scale', Scale, 'Epsilon', Epsilon, 'OperationDimension',OpDimension);
    
                issues = [issues constructionIssue];
            end
        end
    end

    methods(Access=protected)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            % To be fully supported, the input format and first output
            % format must be the same.
            % SupportedONNXLabels
            supportedFormats = ["BSSC", "SSC", "BSSSC", "SSSC", "TBC", "BTC", "BC",...
                "BCSS", "BCSSS", "CSS", "CSSS", "BCT", "TBCSS", "TBCSSS"]; 
            tf = ismember(inputTensorFormat(1), supportedFormats) && ...
                isequal(inputTensorFormat(1), outputTensorFormat(1));

        end
        
        function OpDimension = getOperationDimension(this, inputTensorFormat)
            % The built-in LayerNormalizationLayer uses channel-wise Scale
            % and Offset. It could be translated to built-in LayerNormalizationLayer 
            % only when the pass in Scale and Offset are channel-wise.
            OpDimension = [];
            if ~isscalar(double(initializerSize(this.GraphProtoManager, this.Node.input{2})))
                return;
            end
            axis = this.axis + 1;
            dim = strlength(inputTensorFormat(1));
            if axis <= 0
                axis = dim + axis;
            end
            switch inputTensorFormat(1)
                case {"BSSC", "BSSSC", "BC"}
                    if axis == dim
                        OpDimension = "channel-only";
                    elseif axis == 2
                        OpDimension = "spatial-channel";
                    end
                case {"CSSS", "CSS"}
                    if axis == 1
                        OpDimension = "spatial-channel";
                    end
                case {"BCSS", "BCSSS"}
                    if axis == 2
                        OpDimension = "spatial-channel";
                    end
                case {"TBCSS", "TBCSSS"}
                    if axis == 3
                        OpDimension = "spatial-channel";
                    end
                case {"SSC","SSSC"}
                    if axis == dim
                        OpDimension = "channel-only";
                    elseif axis == 1
                        OpDimension = "spatial-channel";
                    end
                case "TBC"
                    if axis == 3
                        OpDimension = "channel-only";
                    end
                case "BTC"
                    if axis == 3
                        OpDimension = "channel-only";
                    elseif axis == 2
                        OpDimension = "batch-excluded";
                    end
                case "BCT"
                    if axis == 2
                        OpDimension = "channel-only";
                    end
            end
        end

        function tf = inputsInitializers(this,hasBias)
            % To be fully supported, input 2 and optional input 3 must be initializers.
            tf = isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) ...
                && (hasBias == isNodeInputInitializer(this.GraphProtoManager, this.Node, 3));
        end
        
        function data = reshapeData(~, data, NumChannels, inputTensorFormat)
            % Reshape the data to [C 1] [1 1 C] or [1 1 1 C]
            switch inputTensorFormat
                case {'BCSSS','BSSSC','SSSC','CSSS','TBCSSS'}
                    data = reshape(data, [1 1 1 NumChannels]);
                case {'BSSC','BCSS','SSC','CSS','TBCSS'}
                    data = reshape(data, [1 1 NumChannels]);
                otherwise
                    data = reshape(data, [NumChannels 1]);
            end
        end
    end
end