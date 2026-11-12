classdef TranslatorForClip < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Clip operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        % Operator attributes
        max
        min
        
        % Other properties
        OpsetVersion
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "max"  	"FLOAT"     true    Inf
                "min" 	"FLOAT"  	true    -Inf
                });
            [this.max, this.min] = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
            
            % Set the opset
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
            [Min, Max, issues] = getBounds(this);
            if isempty(issues) && (isLabeled(this, inputTensorFormats, outputTensorFormats) || this.GenerateCustomLayers==false)
                [Layer, issues] = constructLayer(this, 'nnet.onnx.layer.ClipLayer', this.LayerName, this.Node, this.LayerName, Min, Max);
            else
                Layer = [];
            end
        end
    end
    
    methods(Access=protected)
        function tf = isLabeled(~, inputTensorFormat, outputTensorFormat)
            tf = ~isequal(inputTensorFormat, "") &&  ~isequal(outputTensorFormat, "");
        end
        
        function [Min, Max, issues] = getBounds(this)
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            Min = this.min; Max = this.max;
            if this.OpsetVersion > 10
                % Min and Max are node inputs starting in Opset 11
                % ONNX Specification requires these to be scalar
                if numel(this.Node.input) > 1
                    if isTensorInitializer(this.GraphProtoManager, this.Node.input{2})
                        Min = initializerRawData(this.GraphProtoManager, this.Node.input{2});
                    else
                        issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                            message('nnet_cnn_onnx:onnx:ClipFromOperator', this.LayerName));
                    end
                end
                
                if numel(this.Node.input) > 2
                    if isTensorInitializer(this.GraphProtoManager, this.Node.input{3})
                        Max = initializerRawData(this.GraphProtoManager, this.Node.input{3});
                    else
                        issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                            message('nnet_cnn_onnx:onnx:ClipFromOperator', this.LayerName));
                    end
                end
                
                % The min and max inputs are optional. If they are empty, we
                % interpret them as skipped and stick to the default value.
                if isempty(Min)
                    Min = -Inf;
                end
                
                if isempty(Max)
                    Max = Inf;
                end
            end
            
        end
    end
end
