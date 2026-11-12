classdef ONNXParameters < deep.internal.sdk.ParameterContainer
    %ONNXParameters An object representing the parameters of an imported ONNX network.
    %
    %   ONNXParameters Properties:
    %
    %     Learnables    - A struct containing all learnable network
    %                     parameters.
    %     Nonlearnables - A struct containing all nonlearnable network
    %                     parameters. For example, convolution strides and
    %                     padding.
    %     State         - A struct containing all network 'state'
    %                     variables, which are updated across multiple
    %                     training batches, for example, batch
    %                     normalization running averages.
    %     NumDimensions - A struct containing the intended number of
    %                     dimensions of each parameter (required by some
    %                     ONNX operations).
    %     NetworkFunctionName - The name of the network function to which
    %                     these parameters apply.
    %
    %   ONNXParameters Methods:
    %     addParameter          - Add a new parameter.
    %     removeParameter       - Remove a parameter.
    %     freezeParameters      - Make one or more Learnable parameters nonlearnable.
    %     unfreezeParameters    - Make one or more nonlearnable parameters Learnable.
    %
    %   ONNXParameters Examples:
    %
    %   (1) Create a ONNXParameters object by importing an ONNX network into a function:
    %       p = importONNXFunction('Alexnet.onnx')
    %
    %   (2) Add parameters:
    %       p = addParameter(p, 'weight1', rand(128,16), 'learnable')
    %       p = addParameter(p, 'weight2', rand(64,32,1), 'Learnable')
    %       p = addParameter(p, 'bias1', rand(16,1), 'nonlearnable')
    %
    %   (3) Assign a new value to a parameter:
    %       p.Learnables.weight1 = zeros(128,16)
    
    %   Copyright 2020-2022 The MathWorks, Inc.
    
    properties (Dependent)
        % A struct containing all learnable network parameters. For
        % example, weights.
        Learnables
        
        % A struct containing all nonlearnable network parameters. For example,
        % convolution strides and padding.
        Nonlearnables
        
        % A struct containing all network 'state' variables, which are
        % updated across multiple training batches. For example, batch
        % normalization running averages.
        State        
    end
    
    properties (SetAccess=protected)
        % A struct containing the intended number of dimensions of each
        % parameter (required by some ONNX operations).
        NumDimensions

        % The name of the model function
        NetworkFunctionName
    end
    
    properties (Hidden, Access=protected)
        LearnablesStruct    = struct;
        NonlearnablesStruct	= struct;
        StateStruct         = struct;
    end
    
    % Constructor is hidden
    methods(Hidden)
        function this = ONNXParameters(networkFunctionName)
            this.NetworkFunctionName = networkFunctionName;
        end
    end
    
    % Hidden static method to create an instance
    methods(Static, Hidden)
        function this = createONNXParameters(networkFunctionName)
            this = ONNXParameters(networkFunctionName);
        end
    end
    
    % Public methods
    methods
        function params = addParameter(params, paramName, paramValue, paramKind, numDimensions)
            %ONNXParameters/addParameter Add an ONNX parameter.
            % params = addParameter(params, paramName, paramValue, paramKind, numDimensions)
            %   adds a new parameter to params.
            %
            % Inputs:
            %   params          - A ONNXParameters object.
            %   paramName       - A string.
            %   paramValue      - A numeric array or string.
            %   paramKind       - One of 'learnable', 'nonlearnable', or 'state'.
            %   numDimensions	- (Optional) The intended number of dimensions
            %                     of paramValue. A nonnegative integer.
            %                   Default: ndims(paramValue)
            if nargin < 5
                numDimensions = ndims(paramValue);
            end
            checkAddArgs(params, paramName,  paramValue, paramKind, numDimensions, nargout);
            switch lower(paramKind)
                case 'learnable'
                    params.LearnablesStruct.(paramName) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(paramValue);
                    params.NumDimensions.(paramName)      = numDimensions;
                case 'nonlearnable'
                    params.NonlearnablesStruct.(paramName) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(paramValue);
                    params.NumDimensions.(paramName)     = numDimensions;
                case 'state'
                    params.StateStruct.(paramName) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(paramValue);
                    params.NumDimensions.(paramName) = numDimensions;
                otherwise
                    assert(false);
            end
        end
        
        function params = removeParameter(params, paramName)
            %ONNXParameters/removeParameter Remove an ONNX parameter.
            % params = removeParameter(params, paramName)
            %   Removes a parameter from params.
            %
            % Inputs:
            %   params  - A ONNXParameters object.
            %   paramName   - A string.
            validateattributes(paramName, {'string','char'}, {'scalartext'}, '', 'paramName');
            if isfield(params.LearnablesStruct, paramName)
                params.LearnablesStruct = rmfield(params.LearnablesStruct, paramName);
                params.NumDimensions = rmfield(params.NumDimensions, paramName);
            elseif isfield(params.NonlearnablesStruct, paramName)
                params.NonlearnablesStruct = rmfield(params.NonlearnablesStruct, paramName);
                params.NumDimensions = rmfield(params.NumDimensions, paramName);
            elseif isfield(params.StateStruct, paramName)
                params.StateStruct = rmfield(params.StateStruct, paramName);
                params.NumDimensions = rmfield(params.NumDimensions, paramName);
            else
                warning(message('nnet_cnn_onnx:onnx:paramNotFound', paramName)); 
            end
            if nargout < 1
                warning(message('nnet_cnn_onnx:onnx:removeOutputNotAssigned'));
            end
        end
        
        function params = freezeParameters(params, paramNames)
            %ONNXParameters/freezeParameters Make some existing Learnable
            %parameters Nonlearnable.
            %
            % params = freezeParameters(params, 'all') sets all
            %   Learnable parameters to be Nonlearnable.
            %
            % params = freezeParameters(params, paramNames), where
            %	paramNames is a string array or cell array of strings, sets
            %	the Learnable parameters in paramNames to be Nonlearnable.
            validateattributes(paramNames, {'string','char','cell'}, {}, '', 'paramNames');
            if isequal(paramNames, "all")
                paramNames = string(fieldnames(params.LearnablesStruct));
            elseif iscellstr(paramNames) || isstring(paramNames) || ischar(paramNames)
                paramNames = string(paramNames);
            else
                error(message('nnet_cnn_onnx:onnx:freezeAndUnfreezeInput', 'freezeParameters'));
            end
            for i = 1:numel(paramNames)
                if ~isfield(params.LearnablesStruct, paramNames(i))
                    error(message('nnet_cnn_onnx:onnx:learnableNotFound', paramNames(i)));
                end
                params.NonlearnablesStruct.(paramNames(i)) = params.LearnablesStruct.(paramNames(i));
                params.LearnablesStruct = rmfield(params.LearnablesStruct, paramNames(i));
            end
            if nargout < 1
                warning(message('nnet_cnn_onnx:onnx:freezeOutputNotAssigned'));
            end
        end
        
        function params = unfreezeParameters(params, paramNames)
            %ONNXParameters/unfreezeParameters Make some existing Nonlearnable
            %parameters Learnable.
            %
            % params = unfreezeParameters(params, 'all') sets all
            %   Nonlearnable parameters to be Learnable.
            %
            % params = unfreezeParameters(params, paramNames), where
            %	paramNames is a string array or cell array of strings, sets
            %	the Nonlearnable parameters in paramNames to be Learnable.
            validateattributes(paramNames, {'string','char','cell'}, {}, '', 'paramNames');
            if isequal(paramNames, "all")
                paramNames = string(fieldnames(params.NonlearnablesStruct));
            elseif iscellstr(paramNames) || isstring(paramNames) || ischar(paramNames)
                paramNames = string(paramNames);
            else
                error(message('nnet_cnn_onnx:onnx:freezeAndUnfreezeInput', 'unfreezeParameters'));
            end
            for i = 1:numel(paramNames)
                if ~isfield(params.NonlearnablesStruct, paramNames(i))
                    error(message('nnet_cnn_onnx:onnx:nonlearnableNotFound', paramNames(i)));
                end
                val = params.NonlearnablesStruct.(paramNames(i));
                if ~isnumeric(val)
                    error(message('nnet_cnn_onnx:onnx:learnablesNumeric'));
                end
                params.LearnablesStruct.(paramNames(i)) = val;
                params.NonlearnablesStruct = rmfield(params.NonlearnablesStruct, paramNames(i));
            end
            if nargout < 1
                warning(message('nnet_cnn_onnx:onnx:unfreezeOutputNotAssigned'));
            end
        end
        
        function S = get.Learnables(this)
            S = this.LearnablesStruct;
        end
        
        function S = get.Nonlearnables(this)
            S = this.NonlearnablesStruct;
        end
        
        function S = get.State(this)
            S = this.StateStruct;
        end
        
        function this = set.Learnables(this, newStruct)
            assert(isstruct(newStruct), message('nnet_cnn_onnx:onnx:mustBeAStruct', 'Learnables'));
            % Make sure field names match
            fields = fieldnames(this.LearnablesStruct);
            newFields = fieldnames(newStruct);
            assert(isequal(sort(fields), sort(newFields)), message('nnet_cnn_onnx:onnx:toAddOrRemoveParams'));
            % Make sure the new numdims are compatible with stored numdims
            for i = 1:numel(fields)
                curNumdims = this.NumDimensions.(fields{i});
                newval = newStruct.(fields{i});
                lastNonsingletonDim = find(size(newval)>1, 1, 'last');
                if ~isempty(lastNonsingletonDim) && curNumdims < lastNonsingletonDim
                    warning(message('nnet_cnn_onnx:onnx:paramDims', fields{i}));
                    this.NumDimensions.(fields{i}) = ndims(newval);
                end
                if ~isnumeric(newval)
                    error(message('nnet_cnn_onnx:onnx:learnablesNumeric'));
                end
                % Convert numeric value to double
                newval = double(newval);
                this.LearnablesStruct.(fields{i}) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(newval);
            end
        end
        
        function this = set.Nonlearnables(this, newStruct)
            assert(isstruct(newStruct), message('nnet_cnn_onnx:onnx:mustBeAStruct', 'Nonlearnables'));
            % Make sure field names match
            fields = fieldnames(this.NonlearnablesStruct);
            newFields = fieldnames(newStruct);
            assert(isequal(sort(fields), sort(newFields)), message('nnet_cnn_onnx:onnx:toAddOrRemoveParams'));
            % Make sure the new numdims are compatible with stored numdims
            for i = 1:numel(fields)
                curNumdims = this.NumDimensions.(fields{i});
                newval = newStruct.(fields{i});
                lastNonsingletonDim = find(size(newval)>1, 1, 'last');
                if ~isempty(lastNonsingletonDim) && curNumdims < lastNonsingletonDim
                    warning(message('nnet_cnn_onnx:onnx:paramDims', fields{i}));
                    this.NumDimensions.(fields{i}) = ndims(newval);
                end
                % Convert numeric value to double
                if isnumeric(newval)
                    newval = double(newval);
                end
                this.NonlearnablesStruct.(fields{i}) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(newval);
            end
        end
        
        function this = set.State(this, newStruct)
            assert(isstruct(newStruct), message('nnet_cnn_onnx:onnx:mustBeAStruct', 'State'));
            % Make sure field names match
            fields = fieldnames(this.StateStruct);
            newFields = fieldnames(newStruct);
            assert(isequal(sort(fields), sort(newFields)), message('nnet_cnn_onnx:onnx:toAddOrRemoveParams'));
            % Make sure the new numdims are compatible with stored numdims
            for i = 1:numel(fields)
                curNumdims = this.NumDimensions.(fields{i});
                newval = newStruct.(fields{i});
                lastNonsingletonDim = find(size(newval)>1, 1, 'last');
                if ~isempty(lastNonsingletonDim) && curNumdims < lastNonsingletonDim
                    warning(message('nnet_cnn_onnx:onnx:paramDims', fields{i}));
                    this.NumDimensions.(fields{i}) = ndims(newval);
                end
                % Convert numeric value to double
                if isnumeric(newval)
                    newval = double(newval);
                end
                this.StateStruct.(fields{i}) = nnet.internal.cnn.onnx.fcn.makeUnlabeledDlarray(newval);
            end
        end
        
    end
    
    methods (Hidden, Access=protected)
        function checkAddArgs(this, name, value, type, numDimensions, nOut)
            validateattributes(name,{'string','char'},{'scalartext'},'','paramName');
            allNames = [fieldnames(this.LearnablesStruct);
                fieldnames(this.NonlearnablesStruct);
                fieldnames(this.StateStruct)];
            if ismember(name, allNames)
                error(message('nnet_cnn_onnx:onnx:paramNameExists', name));
            end
            validateattributes(numDimensions,{'single','double'},{'scalar','nonnegative','integer'},'','numDimensions');
            lastNonsingletonDim = find(size(value)>1, 1, 'last');
            if ~isempty(lastNonsingletonDim) && numDimensions < lastNonsingletonDim
                error(message('nnet_cnn_onnx:onnx:numdimsAdd'));
            end
            validateattributes(type,{'string','char'},{'scalartext'},'','paramKind');
            if ~ismember(lower(type), {'learnable', 'nonlearnable', 'state'})
                error(message('nnet_cnn_onnx:onnx:paramKind'));
            end
            if nOut < 1
                warning(message('nnet_cnn_onnx:onnx:addOutputNotAssigned'));
            end
        end
    end

    % Implement ParameterContainer interface
    methods (Access = protected)
        function this = applyActionToLearnableParameters(this, action)
            learnables = this.LearnablesStruct;
            this.LearnablesStruct = [];
            learnables = apply(action, learnables);
            this.LearnablesStruct = learnables;
        end

        function this = applyActionToAccelerationParameters(this, action)
            learnables = this.LearnablesStruct;
            state = this.StateStruct;
            c = {learnables state};
            if isReadOnly(action)
                apply(action, c);
            else
                this.LearnablesStruct = [];
                this.StateStruct = [];
                c = apply(action, c);
                this.LearnablesStruct = c{1};
                this.StateStruct = c{2};
            end
        end

        function key = getAccelerationCacheKey(this)
            key = {this.Nonlearnables, this.NumDimensions, this.NetworkFunctionName};
        end
    end
    
end
