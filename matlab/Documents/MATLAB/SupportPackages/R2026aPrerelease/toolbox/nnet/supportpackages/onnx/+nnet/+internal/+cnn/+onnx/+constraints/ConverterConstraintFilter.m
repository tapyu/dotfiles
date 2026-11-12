classdef ConverterConstraintFilter < nnet.internal.cnn.analyzer.constraints.Constraint
    % ConverterConstraintFilter  Wraps a constraint to filter out issues
    % unrelated
    
    %   Copyright 2020-2025 The MathWorks, Inc.
    
    properties
        UnderlyingConstraints
        HasMissingOutput
    end
    
    methods
        
        function this = ConverterConstraintFilter(underlyingConstraints)
            this.UnderlyingConstraints = underlyingConstraints;
            this.HasMissingOutput = false;
        end
        
    end
    
    methods

        function testUnderlyingConstraint(this)
            issues = this.UnderlyingConstraints.applyConstraints(this.NetworkAnalyzer);
            missingOutputIssues = iIsMissingOutputIssue(issues);
            this.HasMissingOutput = any(missingOutputIssues);
            
            if this.HasMissingOutput
                % Remove missing output issues.
                issues(missingOutputIssues, :) = [];
            end

            % Now remove any unrelated issues from issues table
            issues = iRemoveUnrelatedIssues(issues);
            this.addIssuesTable(issues);
        end
     
        
    end
    
    methods (Access = private)
        
        function addIssuesTable(this, issues)
            for i=1:size(issues,1)
                this.addIssueWithId(...
                    issues.Severity{i}, ...
                    issues.Type{i}, ...
                    issues.Layers{i}, ...
                    issues.Id{i}, ...
                    issues.Message{i}, ...
                    issues.UserData{i} ...
                );
            end
        end
    end
end

function tf = iIsMissingOutputIssue(issues)
tf = ismember(string(issues.Id), [
    "nnet_cnn:internal:cnn:analyzer:constraints:Architecture:MissingOutputLayer"
    "nnet_cnn:internal:cnn:analyzer:constraints:Connections:UnusedOutputs"
    ]);
end

function issues = iRemoveUnrelatedIssues(issues)
unrelatedIssuesIdx = ismember(string(issues.Id), ...
    ["nnet_cnn:internal:cnn:analyzer:constraints:Architecture:HasUnsupportedLayerInDAG",...
    "nnet_cnn:internal:cnn:layer:GRUProjected:UnsupportedStateActivInDAGNetwork",...
    "nnet_cnn:internal:cnn:layer:GRU:UnsupportedStateActivInDAGNetwork",...
    "nnet_cnn:internal:cnn:layer:BiLSTM:UnsupportedStateActivInDAGNetwork",...
    "nnet_cnn:internal:cnn:layer:LSTM:UnsupportedStateActivInDAGNetwork"]);
issues(unrelatedIssuesIdx, :) = [];
end