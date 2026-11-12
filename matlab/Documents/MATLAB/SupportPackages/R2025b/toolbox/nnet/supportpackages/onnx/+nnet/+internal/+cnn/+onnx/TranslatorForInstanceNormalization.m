classdef TranslatorForInstanceNormalization < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX InstanceNormalization operators into MATLAB layers
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        consumed_inputs
        epsilon
        
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
                "consumed_inputs"	"INTS"      true    []
                "epsilon"           "FLOAT"     true    1e-5
                });
            % Parse the attributes
            [this.consumed_inputs, this.epsilon] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
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
            if supportedFormat(this, inputTensorFormats, outputTensorFormats) && inputsInitializers(this)
                if this.epsilon <= 0
                    Epsilon = 1e-5;
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node, message('nnet_cnn_onnx:onnx:BadEpsilon'))];
                else
                    Epsilon = double(this.epsilon);
                end
                ScaleName       = this.Node.input{2};
                BiasName        = this.Node.input{3};
                NumChannels     = double(initializerSize(this.GraphProtoManager, ScaleName));
                Offset          = reshapeData(single(initializerRawData(this.GraphProtoManager, BiasName)),  NumChannels, inputTensorFormats(1));
                Scale           = reshapeData(single(initializerRawData(this.GraphProtoManager, ScaleName)), NumChannels, inputTensorFormats(1));
                
                [Layer, constructionIssue] = constructLayer(this, 'instanceNormalizationLayer', this.LayerName, this.Node, ...
                    'Name', this.LayerName, 'Offset', Offset, 'Scale', Scale, 'Epsilon', Epsilon);
                
                issues = [issues constructionIssue];
            end
        end
    end
    
    methods(Access=protected)
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            supportedFormats = ["BSSC", "SSC", "BSSSC", "SSSC", "TBC", "BTC", "BC",...
                "BCSS", "BCSSS", "CSS", "CSSS", "BCT", "TBCSS", "TBCSSS"]; 
            % To be fully supported, the input and output format must be
            % the same
            tf = ismember(inputTensorFormat(1), supportedFormats) && ...
                isequal(inputTensorFormat(1), outputTensorFormat);
        end
        
        function tf = inputsInitializers(this)
            % To be fully supported, inputs 2-3
            % must be initializers
            tf = isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) ...
                && isNodeInputInitializer(this.GraphProtoManager, this.Node, 3);
        end        
    end
end

function data = reshapeData(data, NumChannels, inputTensorFormat)
    % Reshape the data to [1 1 C] or [1 1 1 C] depending on whether the
    % input is 2d or 3d
    switch inputTensorFormat
        case {'BCSSS','BSSSC','SSSC','CSSS','TBCSSS'}
            data = reshape(data, [1 1 1 NumChannels]);
        case {'BSSC','BCSS','SSC','CSS','TBCSS'}
            data = reshape(data, [1 1 NumChannels]);
        otherwise
            data = reshape(data, [NumChannels 1]);
    end
end