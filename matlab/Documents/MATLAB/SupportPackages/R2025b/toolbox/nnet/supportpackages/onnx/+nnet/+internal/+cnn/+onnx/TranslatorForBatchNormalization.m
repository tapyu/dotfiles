classdef TranslatorForBatchNormalization < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX BatchNormalization operators into MATLAB layers
    
    % Copyright 2021-2022 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator Attributes
        consumed_inputs
        epsilon
        is_test
        momentum
        spatial
        training_mode
        
        % Other properties
        OpsetVersion
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "consumed_inputs"	"INTS"      true    []
                "epsilon"           "FLOAT"     true    1e-5
                "is_test"           "INT"       true    0
                "momentum"          "FLOAT"     true    0.9
                "spatial"           "INT"      	true    1
                "training_mode"     "INT"       true    0
                });
            % Parse the attributes
            [this.consumed_inputs, this.epsilon, this.is_test, this.momentum, this.spatial, this.training_mode] = ...
                nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
            % Set the opset version
            this.OpsetVersion = this.GraphProtoManager.OpsetVersion;
            
            % Set the layer name
            this.LayerName = this.Node.name;
            
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            Layer = [];
            if numel(this.Node.output) == 1 || ~any(outputUsedInNetwork(this, [2,3])) % The operator may only have a single output to be translated into a layer.
                % Validate the input/output formats
                if ~supportedFormat(this, inputTensorFormats, outputTensorFormats)
                    Layer = [];
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:UnsupportedInputTensorFormat', inputTensorFormats, 'BatchNormalization'));
                    return;
                end
                
                % Inputs 2-5 must be initializers
                if ~inputsInitializers(this)
                    Layer = [];
                    inputListStr = join(["'scale'", "'B'", "'mean'", "'var'"], ',');
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:InputFromInitializer', inputListStr, 'BatchNormalization'));
                    return;
                end
                
                % Translate the layer
                if this.epsilon <= 0
                    Epsilon = 1e-5;
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                        message('nnet_cnn_onnx:onnx:BadEpsilon'))];
                else
                    Epsilon = double(this.epsilon);
                end
                if this.spatial == 0
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node,...
                        message('nnet_cnn_onnx:onnx:PerChannelMean'))];
                end
                
                ScaleName       = this.Node.input{2};
                BiasName        = this.Node.input{3};
                meanName        = this.Node.input{4};
                varName         = this.Node.input{5};
                NumChannels     = double(initializerSize(this.GraphProtoManager, ScaleName));
                TrainedMean     = reshapeData(this, single(initializerRawData(this.GraphProtoManager, meanName)), NumChannels, inputTensorFormats(1));
                TrainedVariance = reshapeData(this, single(initializerRawData(this.GraphProtoManager, varName)), NumChannels, inputTensorFormats(1));
                Offset          = reshapeData(this, single(initializerRawData(this.GraphProtoManager, BiasName)), NumChannels, inputTensorFormats(1));
                Scale           = reshapeData(this, single(initializerRawData(this.GraphProtoManager, ScaleName)), NumChannels, inputTensorFormats(1));
                
                % Set nonpositive variance components to a value below eps('single')
                nonPos = TrainedVariance <= 0;
                if any(nonPos)
                    issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node, message('nnet_cnn_onnx:onnx:BatchNormNegVar', this.LayerName))];
                    TrainedVariance(nonPos) = realmin('single');
                end
                
                [Layer, constructionIssue] = constructLayer(this, 'batchNormalizationLayer', this.LayerName, this.Node, 'Name', this.LayerName, 'Offset', Offset,...
                    'Scale', Scale, 'Epsilon', Epsilon, 'TrainedMean', TrainedMean, ...
                    'TrainedVariance', TrainedVariance);                
                issues = [issues constructionIssue];
            end
        end
    end
    
    methods (Access=protected)
        
        function tf = supportedFormat(~, inputTensorFormat, outputTensorFormat)
            % To be fully supported, the input format and first output
            % format must both be "BC", "BCSS", "BCT" or "BCSSS".
            supportedFormats = ["BCSS", "BCSSS", "BC", "BCT"];
            tf = ismember(inputTensorFormat(1), supportedFormats) && ...
                isequal(inputTensorFormat(1), outputTensorFormat(1));
        end
        
        function tf = inputsInitializers(this)
            % To be fully supported, inputs 2-5
            % must be initializers
            tf = isNodeInputInitializer(this.GraphProtoManager, this.Node, 2) ...
                && isNodeInputInitializer(this.GraphProtoManager, this.Node, 3) ...
                && isNodeInputInitializer(this.GraphProtoManager, this.Node, 4) ...
                && isNodeInputInitializer(this.GraphProtoManager, this.Node, 5);
        end
        
        function data = reshapeData(~, data, NumChannels, inputTensorFormat)
            % Reshape the data to [C 1] [1 1 C] or [1 1 1 C]
            switch inputTensorFormat
                case "BCSSS"
                    data = reshape(data, [1 1 1 NumChannels]);
                case "BC"
                    data = reshape(data, [NumChannels 1]);
                case "BCT"
                    data = reshape(data, [NumChannels 1]);
                otherwise % "BCSS"
                    data = reshape(data, [1 1 NumChannels]);
            end
        end
        
    end
end
