classdef TranslatorForDropout < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Dropout operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        ratio
        seed
        
        % Other properties
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            % Note: Seed does not have a specific default value. The operator spec
            % only says "if not specified we will auto generate one."
            AttributeTable = cell2table({
                "is_test"    "INT"      true    1
                "ratio"      "FLOAT"    true    0.5
                "seed"       "INT"      true    []
                });
            % Parse the attributes
            [~, this.ratio, this.seed] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
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
            % See if the second output ('mask') is present and used in the
            % graph.
            if numel(this.Node.output)>1 && outputUsedInNetwork(this, 2)
                % Support only 1 output
                issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message("nnet_cnn_onnx:onnx:DropoutIsMO"))];
                Layer = [];
                return;
            end
            if inputTensorFormats(1) ~= "" && outputTensorFormats(1)~="" || this.GenerateCustomLayers==false
                % Translate the layer.
                if ~isempty(this.seed)
                    % Seed will be ignored
                    issues = nnet.internal.cnn.onnx.NodeTranslationWarning(this.Node, message("nnet_cnn_onnx:onnx:IgnoredAttributes", "seed", this.Node.op_type));
                end
                % Ratio is an input starting in opset 12.
                Ratio = this.ratio;
                if numel(this.Node.input) > 1
                    if ~isTensorInitializer(this.GraphProtoManager, this.Node.input{2})
                        % If ratio is not in an initializer, it is not supported.
                        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message("nnet_cnn_onnx:onnx:InputFromInitializer", "ratio", this.Node.op_type))];
                        Layer = [];
                        return;
                    else
                        Ratio = double(initializerRawData(this.GraphProtoManager, this.Node.input{2}));
                    end
                end
                [Layer, constructionIssue] = constructLayer(this, 'dropoutLayer', this.LayerName, this.Node, Ratio, 'Name', this.LayerName);
                issues = [issues constructionIssue];
            end
        end
    end
    
    methods(Access=protected)
        function tf = isLabeled(~, inputTensorFormats, outputTensorFormats)
            tf = ~any(strcmp(inputTensorFormats, "")) && ~any(strcmp(outputTensorFormats, ""));
        end
    end
end
