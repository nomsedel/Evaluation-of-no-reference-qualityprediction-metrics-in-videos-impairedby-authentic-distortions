% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto

clear all 
clc

% Se cargan los puntajes objetvos 
MOS=load('D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\NIQE\\mos_cvd_final.mat');

% Se cargan los puntajes objetivos extraidos de la metrica 
load('D:\JAVERIANA\THESIS\RESULTS\NIQE\REPETICION\CVD\average.mat');
MOS=MOS.mos;

% Se toman aleatoriamente un conjunto de prueba y entrenamiento 
[trainInd,valInd,testInd] = dividerand(234,0.8,0,0.2);  

% Se reescalan los puntajes objetivos de NIQE a la escala de la base de datos   
C=rescale(score_average,0,100);

% Se invierte la escala de los puntajes (porque para NIQE y VIIDEO entre mayor puntaje menor calalidad)
C=-C+100;

% Se crean nuevas matrices con los puntajes escogidos aleatoriamente 
Test_DataC1 = score_average(testInd,:);
Test_MOSC1 = MOS (testInd);

% Asignacion de Betas
beta(1)=max(MOS);
beta(2)=min(MOS);
beta(3)=mean(MOS);
beta(4)=1;

opts = statset('nlinfit');
opts.RobustWgtFun = 'welsch';
model=@(beta,F)(beta(1)-beta(2))./(1+ exp(-(F-(beta(3)./(abs(beta(4)))))))+beta(2);
beta0 = nlinfit(Test_DataC1,Test_MOSC1,model,beta,opts);

% Puntajes objetivos despues de realizar la funcion logistica 
objetivos=feval(model,beta0,C);

% Correlacion para obtener los coeficientes de correlacion PLCC, SROCC y RMSE
for i=1:100    
    [trainInd,valInd,testInd] = dividerand(234,0.8,0,0.2);     

    Test_Data = C(trainInd,:);
    Test_MOS = MOS (trainInd);
    
    yfit=corrcoef(Test_Data,Test_MOS);
    pearson(i)=yfit(1,2);
    spearman(i)=corr(Test_Data,Test_MOS,'Type','Spearman');
    RMSE(i) = sqrt(mean((Test_Data-Test_MOS).^2));    
end 
media_pearson=median(pearson);
media_spearman=median(spearman);
media_RMSE=median(RMSE);
std_pearson=std(pearson);
std_spearman=std(spearman);
std_RMSE=std(RMSE);
