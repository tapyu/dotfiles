classdef TranslatorForSum < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Sum operators into MATLAB layers
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(SetAccess = protected)
        LayerName
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
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
            [inputFormats, outputFormats] = propagateMISOBroadcastOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if all([inputTensorFormats outputTensorFormats] ~= "") || this.GenerateCustomLayers==false
                numInputs = numel(this.Node.input);
                areInputsInitializers = isNodeInputInitializer(this.GraphProtoManager, this.Node, 1:numInputs);
                if any(areInputsInitializers)
                    initIdx = find(areInputsInitializers);
                    Layer = [];
                    inputListStr = makeInputList(this, initIdx);
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node,...
                        message('nnet_cnn_onnx:onnx:InputFromInitializer', inputListStr, 'Sum'));
                    return;
                end
                [Layer, issues] = constructLayer(this, 'additionLayer', this.LayerName, this.Node, numInputs, 'Name', this.LayerName);
            end
        end
    end
    
    methods(Access=private)
        function inputListStr = makeInputList(~, initIdx)
            Xs = repmat("X", numel(initIdx), 1);
            Is = num2str(initIdx(:));
            inputListStr = join([Xs Is], '');
            inputListStr = join(inputListStr', ',');
        end
    end
end
