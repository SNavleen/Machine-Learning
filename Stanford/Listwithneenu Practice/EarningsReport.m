theta = 0;
X = 0;
y = 0;

function J = getCost(X, y, theta)
  m = length(y); 
  J = 0;
  sum = 0;

  for i = 1:m
    predicion = getHypothesis(X(i, :)', theta);
    sum += (predicion - y(i)) .^ 2;
  end;

  J = sum / (2 * m);
end

function h = getHypothesis(X, theta)
  h = theta' * X;
end;

%% Pass the X and Y values to plot
function plotData (x, y)
  figure;
  
  plot(x, y, 'rx');
  ylabel('Earnings per month in 1000s');
  xlabel('Months');
endfunction

function [theta] = normalEqn(X, y)
  theta = zeros(size(X, 2), 1);
  theta = pinv(X' * X) * X' * y;
endfunction


%% Clear and Close Figures
clear ; close all; clc

% Step 1
fprintf('Loading data ... \n');

  %% Load the Earnings Report Data
  EarningsReportData = load('EarningsReportData.txt');

  %% Parse the Data based on the year and month columns
  [date, ~, total] = unique(EarningsReportData(:, 1:2), 'rows');
  %% Get the total value earned for the month
  Data = [date, (accumarray(total, EarningsReportData(:, 4)).* 2.5) ./ 100];

  %% Set the X and Y matrix
  X = Data(:, 2);
  Y = Data(:, 3);
  m = length(Y); % number of training examples

fprintf('Finished loading data. Hit enter to continue.\n');
pause;


%% Step 2
fprintf('Ploting data ... \n');

  Y = Y ./ 1000;
  plotData(X, Y)

fprintf('Finsihed ploting. Hit enter to continue.\n');
pause;


%% Step 3
fprintf('Solving with normal equations ...\n');

  X = [ones(m, 1), X];

  %% Calculate the parameters from the normal equation
  theta = normalEqn(X, Y);

  %% Display normal equation's result
  fprintf('Theta computed from the normal equations: \n');
  fprintf(' %f \n', theta);

  %% Plot the linear fit
  hold on;
  plot(X(:,2), X*theta, '-');
  legend('Training data', 'Linear regression');
  hold off 

fprintf('Hit Enter to continue.\n');
pause;

in = input('Enter a month: ');
prediction = [1, in] * theta;
fprintf('For the month of %f, you will earn %f\n', prediction * 1000);
