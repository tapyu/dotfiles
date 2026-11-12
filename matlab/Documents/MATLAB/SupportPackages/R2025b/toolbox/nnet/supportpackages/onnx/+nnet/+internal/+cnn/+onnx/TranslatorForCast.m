classdef TranslatorForCast < nnet.internal.cnn.onnx.OperatorTranslator

    %   Copyright 2021-2024 The MathWorks, Inc.    

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
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            [inputFormats, outputFormats] = propagateSISOPassthroughOp(this, direction, inputFormats, outputFormats);
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            % Supported if both formats are known
            if all([inputTensorFormats outputTensorFormats] ~= "") || this.GenerateCustomLayers==false
                % If the user is importing into a dlnetwork, create
                % identityLayer (which is only supported in dlnetwork). If
                % the user has requested a DAGNetwork, then create the
                % legacy nnet.onnx.layer.IdentityLayer.
                if this.TargetNetwork=="dlnetwork"
                    [Layer, issues] = constructLayer(this, 'identityLayer', this.Node.name, this.Node, 'Name', this.Node.name);
                else % DAGNetwork
                    [Layer, issues] = constructLayer(this, 'nnet.onnx.layer.IdentityLayer', this.Node.name, this.Node, this.Node.name);
                end
            else
                Layer = [];
                issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            end
        end
    end
end
