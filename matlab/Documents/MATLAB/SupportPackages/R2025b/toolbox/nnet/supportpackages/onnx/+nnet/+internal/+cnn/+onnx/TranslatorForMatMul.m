classdef TranslatorForMatMul < nnet.internal.cnn.onnx.OperatorTranslator
    
    % Copyright 2021 The MathWorks, Inc.
    
    properties(Access=protected)
        IsSupported
        
        SupportedLayouts = ["BC", "TBC", "1BC", "T1BC"];
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to empty if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % C=A*B. The only cases we'll propagate and support are when A
            % ends in "C", the MATLAB equivalent starts with a "C", and B
            % is a 2D initializer. In this case, the input1 and output1
            % formats are the same.
            in2 = this.Node.input{2};
            this.IsSupported = ismember(inputFormats(1), this.SupportedLayouts) ...
                && isTensorInitializer(this.GraphProtoManager, in2) ...
                && numel(initializerSize(this.GraphProtoManager, in2)) == 2;
            if this.IsSupported
                [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
            end
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if this.IsSupported
                % Y = A*B. A can have layout ["BC", "TBC", "1BC", "T1BC"]. B is
                % a 2D initializer.
                weightName  = this.Node.input{2};        % Get weight matrix to determine numFilters/numUnits
                weightDim   = initializerSize(this.GraphProtoManager, weightName);              % [X F] rowmajor
                rawWeight   = single(initializerRawData(this.GraphProtoManager, weightName));
                if isempty(rawWeight)
                    issues = nnet.internal.cnn.onnx.NodeTranslationError(this.Node, message("nnet_cnn_onnx:onnx:WeightHasNoInitializer"));
                    return;
                end
                numUnits    = double(weightDim(2));
                % Create FullyConnectedLayer
                [Layer, issues] = constructLayer(this, 'fullyConnectedLayer', this.Node.name, this.Node, numUnits, 'Name', this.Node.name);
                % Convert rowmajor [X F] to colmajor [F X] --> Just reshape to FX
                Layer.Weights = reshape(rawWeight, fliplr(weightDim));    % Now it's colmajor [F X].
                Layer.Bias = zeros(numUnits,1);                           % Now it's colmajor [F 1].
            end
        end
    end
end