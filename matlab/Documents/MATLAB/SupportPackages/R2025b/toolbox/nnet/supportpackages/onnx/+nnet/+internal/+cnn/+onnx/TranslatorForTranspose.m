classdef TranslatorForTranspose < nnet.internal.cnn.onnx.OperatorTranslator
    
    % Copyright 2021-2024 The MathWorks, Inc.
    
    properties
        perm
        NodeFullySupported = false
    end
    
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;

            AttributeTable = cell2table({
                "perm"     "INTS"      true    []
                });
            % Parse the attributes
            this.perm = nnet.internal.cnn.onnx.parseNodeAttributes(this.Node, AttributeTable);
        end
        
        % Given the node and the current input and output formats, update
        % the formats in the specified direction to the extent possible.
        % Direction is either "forward" or "reverse". Each format variable
        % is a string array, where each string is one of the members of the
        % constant 'SupportedONNXLabels' property, or empty. Set a format
        % string to "" if its format cannot be determined.
        function [inputFormats, outputFormats] = propagateTensorFormats(this, direction, inputFormats, outputFormats)
            % Propagation is possible if the result is a defined format,
            % and the S dimensions have not been permuted relative to each
            % other.
            in  = inputFormats;
            out = outputFormats;
            if direction=="forward"
                if in~="" && out==""
                    NumDims = strlength(in);
                    if isempty(this.perm)                   % reverse the label
                        MLperm = NumDims:-1:1;
                    else
                        MLperm = this.perm+1;               % Make the perm origin-1
                    end
                    inChar = char(in);
                    % Check 'S' perms
                    sLocs = strfind(in, 'S');
                    [~,newSLocs] = ismember(sLocs, MLperm);
                    newOut = string(inChar(MLperm));
                    if ismember(newOut, this.SupportedONNXLabels) && issorted(newSLocs)
                        % The new format is supported and the S's are not being reordered
                        outputFormats = newOut;
                        this.NodeFullySupported = true;
                    end
                end
            else
                % backward
                if out~="" && in==""
                    NumDims = strlength(out);
                    if isempty(this.perm)                       % reverse the label
                        MLperm = NumDims:-1:1;
                    else
                        [~,MLperm] = sort(this.perm, 'ascend');	% Compute inverse perm, origin-1
                    end
                    outChar = char(out);
                    % Check 'S' perms
                    sLocs = strfind(out, 'S');
                    [~,newSLocs] = ismember(sLocs, MLperm);
                    newIn = string(outChar(MLperm));
                    if ismember(newIn, this.SupportedONNXLabels) && issorted(newSLocs)
                        % The new format is supported and the S's are not being reordered
                        inputFormats = newIn;
                        this.NodeFullySupported = true;
                    end
                end
            end
        end
        
        % Translate the node into a MATLAB layer, and return any
        % translation issues.
        function [Layer, issues] = translateIntoLayer(this, inputTensorFormats, outputTensorFormats)
            Layer = [];
            issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            if this.NodeFullySupported
                % Reordering the dimensions does not change the meaning of the
                % tensor, so its MATLAB representation is unchanged.
                % If the user is importing into a dlnetwork, create
                % identityLayer (which is only supported in dlnetwork). If
                % the user has requested a DAGNetwork, then create the
                % legacy nnet.onnx.layer.IdentityLayer.
                if this.TargetNetwork=="dlnetwork"
                    [Layer, issues] = constructLayer(this, 'identityLayer', this.Node.name, this.Node, 'Name', this.Node.name);
                else % DAGNetwork
                    [Layer, issues] = constructLayer(this, 'nnet.onnx.layer.IdentityLayer', this.Node.name, this.Node, this.Node.name);
                end
                issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
            end
        end
    end
end