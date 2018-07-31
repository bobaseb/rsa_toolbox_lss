function sigma_est = detect_gauss(residuals)

%blind deconvolution; tries to estimate original gaussian blur, 1D

octaveDivisions = 20;  
numOfOctaves = 5; 
scaleFactor = 2.0^(1.0/octaveDivisions); 
numOfLevels = octaveDivisions*numOfOctaves+1; 
sigma(1) = 1;

for s = 2:numOfLevels 
       sigma(s) = sigma(s-1)*scaleFactor; 
end 

%when 2 gaussians are convolved they produce a new one where their sigmas
%are added together

% residuals = zscore(residuals);
% % 
% for i = 1:length(residuals)
%     if i<length(residuals)
%         residualz(i) = median(residuals(i:i+1));
%     end
% end
% 
% residuals = residualz';

% for i = 1:length(residuals)
%     if i<length(residuals)
%         residualz(i) = median(residuals(i:i+1));
%     end
% end
% 
% residuals = residualz';
% 
%residuals = zscore(residuals);

xmin = min(residuals);
xmax = max(residuals);
a = 1; %gaussian amplitude

%sigma = 4.5;
%g_fun = @gauss;
%proposal = g_fun(xmin:xmax);

error = [];
for s = 1:length(sigma)
    y=[];
    for x = 1:length(residuals) %round(xmin):round(xmax)
        %y = [y a*exp(-residuals(x)^2/(2*sigma(s)^2))];
        y = [y a*exp(-x^2/(2*sigma(s)^2))];
    end
    proposal = y';
    conv_proposal = ifft(fft(proposal) .* fft(residuals));
    
    error = [error sum(abs(residuals - conv_proposal))];
end

% figure(1)
% plot(sigma,error)

dkernel = [-1 1];
d_error = ifft(fft(error') .* fft(dkernel)); %error differences
d_sigma = ifft(fft(sigma') .* fft(dkernel)); %sigma differences

first_d = d_error./d_sigma; %first derivatives

% figure(2)
% plot(sigma,first_d(:,2))

[val,ind]=max(first_d(:,2));

disp(sigma(ind))

sigma_est = sigma(ind);

end

%disp(i)

%gauss = a * exp(-x^2/(2*sigma^2));
%sigma=4.5;
%x=-51;
%(1/sqrt(2*pi*sigma^2))*exp(-(x^2/2*sigma^2))
%function y = gauss(x)
%    a = 1;
%    y = a * exp(-x^2/(2*sigma^2));
%end