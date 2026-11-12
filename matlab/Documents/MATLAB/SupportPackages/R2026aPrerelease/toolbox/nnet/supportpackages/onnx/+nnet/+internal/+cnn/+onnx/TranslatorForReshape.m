classdef TranslatorForReshape < nnet.internal.cnn.onnx.OperatorTranslator
    % A class to translate ONNX Reshape operators into MATLAB layers

    % Copyright 2021-2025 The MathWorks, Inc.

    properties(SetAccess = protected)
        % Operator Attributes
        allowzero
        
        % Other properties
        OpsetVersion
    end

    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
            % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
            AttributeTable = cell2table({
                "allowzero"	"INT"    true    0
                });
            % Parse the attributes
            this.allowzero = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);            

        end

        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % We will do forward propagation in the following case: The
            % first input has a known label that starts with "B", and the
            % 'shape' input is a 2D initializer. In this case, we will
            % propagate forward to "BC".
            if direction=="forward" && startsWith(inputFormats(1),"B") && outputFormats(1)==""
                if isNodeInputInitializer(this.GraphProtoManager, this.Node, 2)
                    shape = initializerRawData(this.GraphProtoManager, this.Node.input{2});
                    if numel(shape)==2
                        outputFormats(1) = "BC";
                    end
                end
            end
        end

        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            % We will import Reshape into a Flatten if: (1) The I->O
            % transformation is a flatten and (2) The Shape argument to
            % Reshape is an initializer (because otherwise we would create
            % a dangling non-output layer which we can't support yet.)
            if isNodeInputInitializer(this.GraphProtoManager, this.Node, 2)
                % BCSS --> BC
                if inputTensorFormats(1)=="BCSS" && outputTensorFormats=="BC" && ~this.allowzero
                    [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.FlattenInto2dLayer', this.Node.name, this.Node, this.Node.name);
                    issues = [issues constructionIssue];
                end
                % BCSSS --> BC
                if inputTensorFormats(1)=="BCSSS" && outputTensorFormats=="BC" && ~this.allowzero
                    [Layer, constructionIssue] = constructLayer(this, 'nnet.onnx.layer.Flatten3dInto2dLayer', this.Node.name, this.Node, this.Node.name);
                    issues = [issues constructionIssue];
                end
                % The reshape operation is translated into custom layer
                % with second input as a Initializer
                if isempty(Layer)
                    this.NeedVerifyBatchSize = true;
                end
            end
        end
    end
end
