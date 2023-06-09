
using Arpack, SparseArrays, LinearAlgebra
# using ExpmV
using NLsolve
using Plots
using Distributions, Random
using JLD
using Combinatorics
using LaTeXStrings

################ basis functions for vNE

function createH!(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    matH .= 0.0
    Depsilon = W/(K-1)
    matH[1,1] = 0.0 # epsilon for the system, probably 0

    for kk = 1:K
        matH[1+kk,1+kk] = (kk-1)*Depsilon - W/2 # epsilon for the bath L
        matH[1+kk,1] = sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        matH[1+K+kk,1] = sqrt(GammaR*Depsilon/(2*pi)) # tunnel with the bath R
    end
    matH[K+2:end,K+2:end] = matH[2:K+1,2:K+1] # epsilon for the bath R

    matH .= matH + matH' - spdiagm(diag(matH))

end

function funbetamu!(F,x,epsilon::Vector{Float64},Ene::Float64,Np::Float64)

    # x[1] = beta, x[2] = mu
    # Depsilon = W/(K-1)

    for kk = 1:length(epsilon)
        # epsilonk = (kk-1)*Depsilon - W/2
        if kk == 1
           F[1] = 1.0/(exp((epsilon[kk]-x[2])*x[1])+1.0)*epsilon[kk]
           F[2] = 1.0/(exp((epsilon[kk]-x[2])*x[1])+1.0)
        else
           F[1] += 1.0/(exp((epsilon[kk]-x[2])*x[1])+1.0)*epsilon[kk]
           F[2] += 1.0/(exp((epsilon[kk]-x[2])*x[1])+1.0)
        end
    end

    F[1] = F[1] - Ene
    F[2] = F[2] - Np

end

function funbetamu2!(F,x,epsilon::Vector{Float64},Ene::Float64,Np::Float64,Cgg::Vector{Float64},matCgg::Matrix{Float64},val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64})

    # x[1] = beta, x[2] = mu

    # Cgg = zeros(Float64,K*2+1)
    Cgg .= 0.0
    for kk = 1:length(val_matH)
        Cgg[kk] = 1.0/(exp((val_matH[kk]-x[2])*x[1])+1.0)
    end
    matCgg .= diagm(Cgg) # f basis
    matCgg .= vec_matH*matCgg*invvec_matH # c basis
    Cgg .= diag(matCgg)

    F[1] = sum(Cgg.*epsilon) - Ene
    F[2] = sum(Cgg) - Np

end

function funbetamu_uni!(F,x,K::Int64,W::Int64,Ene::Float64,Np::Float64)

    # x[1] = beta, x[2] = mu
    Depsilon = W/(K-1)

    for kk = 1:K
        epsilonk = (kk-1)*Depsilon - W/2
        if kk == 1
           F[1] = 1.0/(exp((epsilonk-x[2])*x[1])+1.0)*epsilonk
           F[2] = 1.0/(exp((epsilonk-x[2])*x[1])+1.0)
        else
           F[1] += 1.0/(exp((epsilonk-x[2])*x[1])+1.0)*epsilonk
           F[2] += 1.0/(exp((epsilonk-x[2])*x[1])+1.0)
        end
    end

    F[1] = F[1] - Ene
    F[2] = F[2] - Np

end

function funeffectivebetamu(epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64)

    sol = nlsolve((F,x) ->funbetamu!(F,x,epsilon,Ene,Np), [beta0; mu0])
    return sol.zero

end

function funeffectivebetamu2(epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64,Cgg::Vector{Float64},matCgg::Matrix{Float64},val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64})

    sol = nlsolve((F,x) ->funbetamu2!(F,x,epsilon,Ene,Np,Cgg,matCgg,val_matH,vec_matH,invvec_matH), [beta0; mu0])
    return sol.zero

end

function funeffectivebetamu_uni(K::Int64,W::Int64,Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64)

    sol = nlsolve((F,x) ->funbetamu_uni!(F,x,K,W,Ene,Np), [beta0; mu0])
    return sol.zero

end

function fun_fluctuDeltaepsilon(K::Int64,W::Int64,numvari::Int64)

    if numvari <= 2
       error("fluctuations may come across the above or blow level. Take a bigger numvari such as 4.")
    end

    epsilonL = zeros(Float64,K)
    epsilonR = zeros(Float64,K)

    epsilonL .= LinRange(-W/2,W/2,K)
    epsilonR .= LinRange(-W/2,W/2,K)
    Depsilon = W/(K-1)
    flucutu0 = Depsilon/numvari

    for kk = 1:K

        flucutu = flucutu0*rand(Uniform(-1,1))
        epsilonL[kk] = epsilonL[kk] + flucutu

        flucutu = flucutu0*rand(Uniform(-1,1))
        epsilonR[kk] = epsilonR[kk] + flucutu

    end

    return epsilonL, epsilonR

end

function fun_randomDeltaepsilon(K::Int64,W::Int64)

    epsilonL = zeros(Float64,K)
    epsilonR = zeros(Float64,K)

    epsilon0 = rand(Uniform(-W/2, W/2), K)
    sort!(epsilon0)
    epsilonL .= epsilon0

    # epsilonR .= epsilon0

    epsilon0 = rand(Uniform(-W/2, W/2), K)
    sort!(epsilon0)
    epsilonR .= epsilon0

    return epsilonL, epsilonR

end

function fun_gaussianDeltaepsilon(K::Int64,W::Int64,numvari::Int64)

    epsilonL = zeros(Float64,K)
    epsilonR = zeros(Float64,K)

    epsilonL[1] = -W/2
    epsilonR[1] = -W/2

    epsilonL[K] = W/2
    epsilonR[K] = W/2

    Depsilon = W/(K-1)
    distri = Normal(Depsilon, Depsilon/numvari)

    for kk = 2:K-1

        trdistri = truncated(distri, 0.0, epsilonL[K]-epsilonL[kk-1])
        epsilonL[kk] = rand(trdistri) + epsilonL[kk-1]

        trdistri = truncated(distri, 0.0, epsilonR[K]-epsilonR[kk-1])
        epsilonR[kk] = rand(trdistri) + epsilonR[kk-1]

    end

    return epsilonL, epsilonR

end

function fun_equalDeltaepsilon(K::Int64,W::Int64)

    epsilonL = zeros(Float64,K)
    epsilonR = zeros(Float64,K)

    epsilonL[1] = -W/2
    epsilonR[1] = -W/2

    epsilonL[K] = W/2
    epsilonR[K] = W/2

    Depsilon = W/(K-1)
    # distri = Normal(Depsilon, Depsilon/numvari)

    for kk = 2:K-1

        # trdistri = truncated(distri, 0.0, epsilonL[K]-epsilonL[kk-1])
        # epsilonL[kk] = rand(trdistri) + epsilonL[kk-1]
        epsilonL[kk] = (kk-1)*Depsilon - W/2

        # trdistri = truncated(distri, 0.0, epsilonR[K]-epsilonR[kk-1])
        # epsilonR[kk] = rand(trdistri) + epsilonR[kk-1]
        epsilonR[kk] = (kk-1)*Depsilon - W/2

    end

    return epsilonL, epsilonR

end

function createH_Deltaepsilon!(K::Int64,W::Int64,numvari::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    Depsilon = W/(K-1)

    if numvari == 0
       println("Equally spacing")
       epsilonL, epsilonR = fun_equalDeltaepsilon(K,W)
    elseif numvari == 1
       println("Random spacing")
       epsilonL, epsilonR = fun_randomDeltaepsilon(K,W)
    else
       println("Fluctuating from equal spacing")
       epsilonL, epsilonR = fun_fluctuDeltaepsilon(K,W,numvari)
    # else
       # error("Input numvari correctly")
    end

    matH[1,1] = 0.0 # epsilon for the system, probably 0

    for kk = 1:K

        matH[1+kk,1+kk] = epsilonL[kk] #(kk-1)*Depsilon - W/2 # epsilon for the bath L
        matH[1+K+kk,1+K+kk] = epsilonR[kk] # epsilon for the bath R

        matH[1+kk,1] = sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        matH[1+K+kk,1] = sqrt(GammaR*Depsilon/(2*pi)) # tunnel with the bath R

        # if kk <= K-1
        #    matH[1+kk,1] = sqrt(GammaL*(epsilonL[kk+1]-epsilonL[kk])/(2*pi)) #sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        #    matH[1+K+kk,1] = sqrt(GammaR*(epsilonR[kk+1]-epsilonR[kk])/(2*pi)) # tunnel with the bath R
        # elseif kk == K
        #    matH[1+kk,1] = sqrt(GammaL*(epsilonL[1]-epsilonL[kk])/(2*pi))
        #    matH[1+K+kk,1] = sqrt(GammaR*(epsilonR[1]-epsilonR[kk])/(2*pi))
        # end

        # if kk >= 2 && kk <= K-1
        #    matH[1+kk,1] = sqrt(GammaL*(epsilonL[kk+1]-epsilonL[kk-1])/2/(2*pi)) #sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        #    matH[1+K+kk,1] = sqrt(GammaR*(epsilonR[kk+1]-epsilonR[kk-1])/2/(2*pi)) # tunnel with the bath R
        # elseif kk == 1
        #    matH[1+kk,1] = sqrt(GammaL*(epsilonL[kk+1]-epsilonL[kk])/(2*pi))
        #    matH[1+K+kk,1] = sqrt(GammaR*(epsilonR[kk+1]-epsilonR[kk])/(2*pi))
        # elseif kk == K
        #    matH[1+kk,1] = sqrt(GammaL*(epsilonL[kk]-epsilonL[kk-1])/(2*pi))
        #    matH[1+K+kk,1] = sqrt(GammaR*(epsilonR[kk]-epsilonR[kk-1])/(2*pi))
        # end

    end
    # matH[K+2:end,K+2:end] = matH[2:K+1,2:K+1] # epsilon for the bath R

    matH .= matH + matH' - spdiagm(diag(matH))

end

function createH_fluctuatedt!(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    matH .= 0.0
    Depsilon = W/(K-1)
    tunnelL = sqrt(GammaL*Depsilon/(2*pi))
    tunnelR = sqrt(GammaR*Depsilon/(2*pi))

    matH[1,1] = 0.0 # epsilon for the system, probably 0

    for kk = 1:K
        matH[1+kk,1+kk] = (kk-1)*Depsilon - W/2 # epsilon for the bath L
        flucutu = tunnelL*rand(Uniform(-1,1))*t_flu
        matH[1+kk,1] = tunnelL + flucutu # tunnel with the bath L
        flucutu = tunnelR*rand(Uniform(-1,1))*t_flu
        matH[1+K+kk,1] = tunnelR + flucutu # tunnel with the bath R
    end
    matH[K+2:end,K+2:end] = matH[2:K+1,2:K+1] # epsilon for the bath R

    matH .= matH + matH' - spdiagm(diag(matH))

end

function createH_differentKLKR!(KL::Int64,KR::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    matH .= 0.0
    DepsilonL = W/(KL-1)
    DepsilonR = W/(KR-1)
    tunnelL = sqrt(GammaL*DepsilonL/(2*pi)) #GammaL
    tunnelR = sqrt(GammaR*DepsilonR/(2*pi)) #GammaR

    matH[1,1] = 0.0 # epsilon for the system

    for kk = 1:KL
        matH[1+kk,1+kk] = (kk-1)*DepsilonL - W/2 # epsilon for the bath L
        matH[1+kk,1] = tunnelL # tunnel with the bath L
    end
    for kk = 1:KR
        matH[1+KL+kk,1+KL+kk] = (kk-1)*DepsilonR - W/2 # epsilon for the bath R
        matH[1+KL+kk,1] = tunnelR # tunnel with the bath R
    end

    matH .= matH + matH' - spdiagm(diag(matH))

end

function globalGibbsstate(val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64},beta::Float64,mu::Float64)

    # global Gibbs state
    sizetot = length(val_matH)
    Cgg = zeros(Float64,sizetot)
    for kk = 1:sizetot
        Cgg[kk] = 1.0/(exp((val_matH[kk]-mu)*beta)+1.0)
    end
    Cgg = diagm(Cgg)
    Cgg .= vec_matH*Cgg*invvec_matH

    return Cgg

end

function heatcapacityeff(C0::Vector{Float64},epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # C0 = zeros(Float64,K)
    C0 .= 0.0
    for kk = 1:length(C0)
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
    end

    varH = sum(epsilon.^2 .*(C0.*(1.0.-C0)))
    varN = sum(C0.*(1.0.-C0))
    varHN = sum(epsilon.*(C0.*(1.0.-C0)))

    dUdbeta = -varH + mu*varHN
    dUdmu = beta*varHN
    dNdbeta = mu*varN - varHN
    dNdmu = beta*varN

    return [-beta^2*dUdbeta dUdmu; -beta^2*dNdbeta dNdmu]

end

function Evariance_Gibbs(C0::Vector{Float64},epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # C0 = zeros(Float64,K)
    C0 .= 0.0
    for kk = 1:length(C0)
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
    end

    varH = sum(epsilon.^2 .*(C0.*(1.0.-C0)))

    return varH

end

function Nvariance_Gibbs(C0::Vector{Float64},epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # C0 = zeros(Float64,K)
    C0 .= 0.0
    for kk = 1:length(C0)
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
    end

    varN = sum(C0.*(1.0.-C0))

    return varN

end

function compute_vNEpi(epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # vNE[beta(t)]-vNE[beta(0)] = int dt dQdt*beta(t)

    # correlation matrix
    K = length(epsilon)
    C0 = zeros(Float64,K)
    vNEpi = 0

    for kk = 1:K
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
        if C0[kk] > 0 && C0[kk] < 1
           vNEpi += -C0[kk]*log(C0[kk]) - (1.0-C0[kk])*log(1.0-C0[kk])
        end
    end

    # vNEpi = - sum(C0.*log.(C0)) - sum((1.0 .- C0).*log.(1.0 .- C0))

    return vNEpi

end

function distribute_timepoint(Nt::Int64,ti::Float64,tf::Float64)

    if ti == 0.0
       ti = 1e-4
    end

    ti_log10 = log10(ti)
    tf_log10 = log10(tf)

    time_log10 = LinRange(ti_log10,tf_log10,Nt)
    time = 10.0.^(time_log10)

    return time

end

function plot_efftem(effparaL,effparaR,effpara0,Nt,Gamma,time)

    plot(log10.(Gamma*time),real(effparaL[:,1]),lw=4,label=L"\beta_{L,t}^*",palette=:reds,framestyle = :box)
    plot!(log10.(Gamma*time),real(effparaR[:,1]),lw=4,label=L"\beta_{R,t}^*",palette=:reds)
    plot!(log10.(Gamma*time),real(effpara0[1]*ones(Nt)),lw=2,color=:black,ls=:dash,label=L"\beta_{ref}^*")

    # ylims!((0,1.05))
    xlims!((-2.5,7))
    plot!(xlabel=L"log_{10}\Gamma t")
    plot!(aspect_ratio=6.0)

end

function plot_effchem(effparaL,effparaR,effpara0,Nt,Gamma,time)

    plot(log10.(Gamma*time),real(effparaL[:,2]),lw=4,label=L"\mu_{L,t}^*",palette=:blues,framestyle = :box)
    plot!(log10.(Gamma*time),real(effparaR[:,2]),lw=4,label=L"\mu_{R,t}^*",palette=:blues)
    plot!(log10.(Gamma*time),real(effpara0[2]*ones(Nt)),lw=2,color=:black,ls=:dash,label=L"\mu_{ref}^*")

    # ylims!((0,2.05))
    xlims!((-2.5,7))
    plot!(xlabel=L"log_{10}\Gamma t")
    plot!(aspect_ratio=3.0)

end

function plot_Drelratio(Drel_rhoL_piL_ratio,Drel_rhoR_piR_ratio,Gamma,time)

    plot(log10.(Gamma*time),Drel_rhoL_piL_ratio,lw=4,label=L"L",palette=:greens,framestyle = :box)
    plot!(log10.(Gamma*time),Drel_rhoR_piR_ratio,lw=4,label=L"R",palette=:greens,framestyle = :box)

    xlims!((-2.5,7))

    plot!(xlabel=L"log_{10}\Gamma t")

end

function plot_sigmas(sigma,sigma_c2,I_SE,I_B,I_L,I_R,Gamma,time)

    plot(log10.(Gamma*time[2:end]),real(log10.(sigma[2:end])),label=L"\sigma",color=:grey,lw=6,framestyle = :box)
    plot!(log10.(Gamma*time[2:end]),real(log10.(sigma_c2[2:end])),label=L"\Sigma",color=:black,lw=5)
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_SE[2:end]+I_L[2:end]+I_R[2:end]+I_B[2:end])),label=L"I_{SB}+I_{B}+I_L+I_R",color=:red,lw=4)
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_SE[2:end]+I_L[2:end]+I_R[2:end])),label=L"I_{SB}+I_{B}",color=:blue,lw=3)
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_SE[2:end])),label=L"I_{SB}",color=:green,lw=2)

    # plot(log10.(Gamma*time[2:end]),real(sigma[2:end]),label=L"\sigma",color=:grey,lw=5)
    # plot!(log10.(Gamma*time[2:end]),real(sigma_c2[2:end]),label=L"\Sigma",color=:black,lw=5)
    # plot!(log10.(Gamma*time[2:end]),real(I_SE[2:end]+I_L[2:end]+I_R[2:end]+I_B[2:end]),label=L"I_{SB}+I_{B}+I_L+I_R",color=:red,lw=5)
    # plot!(log10.(Gamma*time[2:end]),real(I_SE[2:end]+I_L[2:end]+I_R[2:end]),label=L"I_{SB}+I_{B}",color=:blue,lw=5)
    # plot!(log10.(Gamma*time[2:end]),real(I_SE[2:end]),label=L"I_{SB}",color=:green,lw=5)

    plot!(legend=:none)
    # plot!(legend=:outerright)
    # ylims!((0,1.1))
    # plot!(xlabel=L"log_{10}\Gamma t")
    xlims!((-2.5,7))
    ylims!((-3,3.5))

end

function plot_sigmas_sub(I_SE,I_B,I_L,I_R,Gamma,time,Drelnuk,Drelpinuk2)

    plot(log10.(Gamma*time[2:end]),real(log10.(I_SE[2:end])),label=L"I_{SB}",lw=2,framestyle = :box)
    
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_R[2:end])),label=L"I_{R}",lw=4)
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_L[2:end])),label=L"I_{L}",lw=6)
 
    plot!(log10.(Gamma*time[2:end]),real(log10.(I_B[2:end])),label=L"I_{B}",lw=8)

    # plot!(log10.(Gamma*time[2:end]),real(log10.(Drelnuk[2:end])),label=L"Drelnuk",lw=10)
    # plot!(log10.(Gamma*time[2:end]),real(log10.(Drelpinuk2[2:end])),label=L"Drelpinuk",lw=12)

    xlims!((-2.5,7))
    ylims!((-3,3.5))

    plot!(legend=:none)
    # plot!(legend=:outerright)
    # ylims!((0,1.1))
    # plot!(xlabel=L"log_{10}\Gamma t")

end

function plot_sigmas2(sigma,sigma_c2,I_SE,I_B,I_L,I_R,Gamma,time)

    plot(log10.(Gamma*time[2:end]),real(log10.(sigma_c2[2:end])),label=L"\Sigma",color=:black,lw=5)
    # plot!(log10.(Gamma*time[2:end]),real(log10.(I_SE[2:end])),label=L"I_{SB}",color=:red,lw=2)
    # plot!(log10.(Gamma*time[2:end]),real(log10.(I_B[2:end])),label=L"I_{B}",color=:blue,lw=2)
    # plot!(log10.(Gamma*time[2:end]),real(log10.(I_L[2:end])),label=L"I_{L}",color=:green,lw=2)
    # plot!(log10.(Gamma*time[2:end]),real(log10.(I_R[2:end])),label=L"I_{R}",ls=:dash,color=:orange,lw=2)
    plot!(log10.(Gamma*time[2:end]),real(log10.(sigma_c2[2:end]-(I_SE[2:end]+I_L[2:end]+I_R[2:end]+I_B[2:end]))),label=L"D_{env}",color=:grey,lw=4)

    plot!(legend=:none)
    plot!(legend=:topleft)
    # ylims!((0,1.1))
    # plot!(xlabel=L"log_{10}\Gamma t")

end

function plot_correlations(I_SE,I_B,I_L,I_R,Drelnuk,Drelpinuk2,Gamma,time)

    tt0 = 0
    for tt = 1:length(time)
        if Gamma*time[tt] > 10^4
           tt0 = tt
           break
        end
    end

    # tt1 = length(time)
    tt1 = 0
    for tt = 1:length(time)
        if Gamma*time[tt] > 10^6
           tt1 = tt-1
           break
        end
    end

    aveI_SE = mean(I_SE[tt0:tt1])
    aveI_B = mean(I_B[tt0:tt1])
    aveI_L = mean(I_L[tt0:tt1])
    aveI_R = mean(I_R[tt0:tt1])
    aveDrelnuk = mean(Drelnuk[tt0:tt1])
    aveDrelpinuk2 = mean(Drelpinuk2[tt0:tt1])

    return aveI_SE, aveI_B, aveI_L, aveI_R, aveDrelnuk, aveDrelpinuk2

end

function plot_correlations2(ind::Int64)

    # rangeX1 = [10^(-1), 10^(-0.5), 1, 2]
    #
    # rangeI_SE1 = [0.07039043163194947, 0.06939321343105473, 0.24232283080188594, 0.42571395145178625]
    # rangeI_B1 = [27.695798372836663, 27.9737337436907, 27.978890852795825, 26.455853375912252]
    # rangeI_L1 = [1.3894858277714415, 4.167210825994144, 8.24892347994511, 8.402646910603945]
    # rangeI_R1 = [1.3864335726091386, 4.169159952802997, 8.259837107613269, 8.37941564266719]
    # rangeDrelnuk1 = [82.07511366963712, 76.91360891907841, 68.50041414241544, 67.4549638504259]
    # rangeDrelpinuk21 = [24.535105856606243, 18.641304228876237, 10.029520948314348, 10.973347588995937]
    #
    # rangeX2 = [1, 10^(0.5), 10, 20]
    #
    # rangeI_SE2 = [0.2979886297560763, 0.5444986327342305, 0.7949759104180926, 0.8302338966313255]
    # rangeI_B2 = [109.76900998189733, 110.13981938673446, 113.74127413628655, 108.18397386410521]
    # rangeI_L2 = [3.0675812346972022, 13.843696442697482, 29.112127945622696, 30.484681425740355]
    # rangeI_R2 = [3.06824773371206, 13.863791459748088, 28.771738523675683, 30.315287521602265]
    # rangeDrelnuk2 = [619.072615119327, 601.5950950264156, 550.262664717758, 560.6026607584242]
    # rangeDrelpinuk22 = [105.4761129317653, 83.25679523114287, 48.10283479922594, 52.426797448746605]

    # rangeX1 = [10^(-1), 10^(-0.5), 1]
    #
    # rangeI_SE1 = [0.07039043163194947, 0.06939321343105473, 0.24232283080188594]
    # rangeI_B1 = [27.695798372836663, 27.9737337436907, 27.978890852795825]
    # rangeI_L1 = [1.3894858277714415, 4.167210825994144, 8.24892347994511]
    # rangeI_R1 = [1.3864335726091386, 4.169159952802997, 8.259837107613269]
    # rangeDrelnuk1 = [82.07511366963712, 76.91360891907841, 68.50041414241544]
    # rangeDrelpinuk21 = [24.535105856606243, 18.641304228876237, 10.029520948314348]
    #
    # rangeX2 = [1, 10^(0.5), 10]
    #
    # rangeI_SE2 = [0.2979886297560763, 0.5444986327342305, 0.7949759104180926]
    # rangeI_B2 = [109.76900998189733, 110.13981938673446, 113.74127413628655]
    # rangeI_L2 = [3.0675812346972022, 13.843696442697482, 29.112127945622696]
    # rangeI_R2 = [3.06824773371206, 13.863791459748088, 28.771738523675683]
    # rangeDrelnuk2 = [619.072615119327, 601.5950950264156, 550.262664717758]
    # rangeDrelpinuk22 = [105.4761129317653, 83.25679523114287, 48.10283479922594]
    #
    # rangeX3 = [10^(-1), 10^(-0.5), 1]
    #
    # rangeI_B3 = [0.688409654942454, 3.7017182438945015, 27.978890852795825]

    rangeX1 = [10^(-1), 10^(-0.5), 1]

    # rangeI_SE1 = [0.07038047692093213, 0.06994065978853194, 0.24019928094819132]
    # rangeI_B1 = [27.687192873879265, 28.12196600216295, 28.283741215039747]
    # rangeI_L1 = [1.374518543575187, 4.036295446926725, 8.133505207947545]
    # rangeI_R1 = [1.3707371149068341, 4.041448350451924, 8.143328547658582]
    # rangeDrelnuk1 = [81.79663214213411, 77.94196462069881, 68.36445316485634]
    # rangeDrelpinuk21 = [24.51130952511794, 18.59507274072866, 9.493944993572612]

    rangeI_SE1 = [0.07646022494835801, 0.08425953117145635, 0.149954191052583]
    rangeI_B1 = [39.60936225102345, 40.02205727397935, 39.949254328993796]
    rangeI_L1 = [0.8564317488626497, 2.471978773211604, 9.078405475177787]
    rangeI_R1 = [0.8633363112711249, 2.4777445400008427, 9.063887180786216]
    rangeDrelnuk1 = [127.10717098889609, 120.922870317337, 107.00212556167828]
    rangeDrelpinuk21 = [36.95847422579063, 33.13860241162712, 19.742713777572522]

    rangeX2 = [1, 10^(0.5), 10]

    rangeI_SE2 = [0.29686462972105854, 0.5458931380743579, 0.7990841873353716]
    rangeI_B2 = [109.3897713416301, 110.03373469917939, 113.86371255423953]
    rangeI_L2 = [2.9649798446817917, 12.750950097422875, 26.402254035214476]
    rangeI_R2 = [2.949083751998387, 12.84684213651656, 26.68543866713233]
    rangeDrelnuk2 = [616.6597866583917, 611.0462337230009, 569.2684134223132]
    rangeDrelpinuk22 = [105.66015921304154, 84.4980039215517, 50.90355045563871]

    if ind == 1
       plot(log10.(rangeX1),rangeI_SE1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_SE2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_SE2,color=:blue,markershapes=:rect,ms=8)
       ylims!((0,1))
    end

    if ind == 2
       plot(log10.(rangeX1),rangeI_B1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_B1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_B2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_B2,color=:blue,markershapes=:rect,ms=8)
       # ylims!((-0.01,0.11))
    end

    if ind == 3
       plot(log10.(rangeX1),rangeI_L1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_L1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_L2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_L2,color=:blue,markershapes=:rect,ms=8)
       # ylims!((0,0.04))
    end

    if ind == 4
       plot(log10.(rangeX1),rangeI_R1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_R1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_R2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_R2,color=:blue,markershapes=:rect,ms=8)
       # ylims!((0,0.04))
    end

    if ind == 5
       plot(log10.(rangeX1),rangeDrelpinuk21,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeDrelpinuk21,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeDrelpinuk22,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeDrelpinuk22,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 6
       plot(log10.(rangeX1),rangeDrelnuk1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeDrelnuk1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeDrelnuk2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeDrelnuk2,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 7
       plot(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_SE2+rangeI_B2+rangeI_L2+rangeI_R2+rangeDrelpinuk22,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_SE2+rangeI_B2+rangeI_L2+rangeI_R2+rangeDrelpinuk22,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 8
       plot(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_SE2+rangeI_B2+rangeI_L2+rangeI_R2+rangeDrelnuk2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_SE2+rangeI_B2+rangeI_L2+rangeI_R2+rangeDrelnuk2,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 9
       plot(log10.(rangeX1),rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_L2+rangeI_R2+rangeDrelpinuk22,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_L2+rangeI_R2+rangeDrelpinuk22,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 10
       plot(log10.(rangeX1),rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:red,lw=5)
       scatter!(log10.(rangeX1),rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:red,ms=8)
       plot!(log10.(rangeX2),rangeI_L2+rangeI_R2+rangeDrelnuk2,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX2),rangeI_L2+rangeI_R2+rangeDrelnuk2,color=:blue,markershapes=:rect,ms=8)
    end

    # plot!(xlabel=L"log_{10}\Gamma t")
    plot!(legend=:none)
    plot!(size=(400,400))

end

function plot_correlations3(ind::Int64)

    # KL=1001, KR=999

    rangeX1 = [1.0, sqrt(10), 5, 8, 10]

    rangeI_SE1 = [0.2700265034560173, 0.5593019137449065, 0.6750499261740152, 0.7604469952134212, 0.8013037387866352]
    rangeI_B1 = [50.9040071968248, 125.6640471482019, 131.58751723662346, 118.42975788943858, 108.69956839611972]
    rangeI_L1 = [2.2641358437554553, 11.367261078792273, 20.732373707936166, 28.086250355927582, 31.09928806697551]
    rangeI_R1 = [1.2820254818286667, 6.42796324091486, 14.98216744455472, 25.763743070895902, 30.18334327288159]
    rangeDrelnuk1 = [70.90234210582268, 273.95624415163667, 418.3749093283321, 534.1528528767477, 560.8428137064682]
    rangeDrelpinuk21 = [34.063524887443684, 46.59327144182832, 47.77443058928806, 48.93522099746331, 48.29375444435278 ]

    if ind == 1
       plot(log10.(rangeX1),rangeI_SE1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1,color=:blue,markershapes=:rect,ms=8)
       # ylims!((0,1))
    end

    if ind == 2
       plot(log10.(rangeX1),rangeI_B1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_B1,color=:blue,markershapes=:rect,ms=8)
       # ylims!((-0.01,0.11))
    end

    if ind == 3
       plot(log10.(rangeX1),rangeI_L1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_L1,color=:blue,markershapes=:rect,ms=8)
       # ylims!((0,0.04))
    end

    if ind == 4
       plot(log10.(rangeX1),rangeI_R1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_R1,color=:blue,markershapes=:rect,ms=8)
       # ylims!((0,0.04))
    end

    if ind == 5
       plot(log10.(rangeX1),rangeDrelpinuk21,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeDrelpinuk21,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 6
       plot(log10.(rangeX1),rangeDrelnuk1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeDrelnuk1,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 7
       plot(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelpinuk21,color=:blue,markershapes=:rect,ms=8)
    end

    if ind == 8
       plot(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:blue,markershapes=:rect,lw=5)
       scatter!(log10.(rangeX1),rangeI_SE1+rangeI_B1+rangeI_L1+rangeI_R1+rangeDrelnuk1,color=:blue,markershapes=:rect,ms=8)
    end

    # plot!(xlabel=L"log_{10}\Gamma t")
    plot!(legend=:none)
    # plot!(size=(400,400))

end

function plot_Fnorm_matC(Fnorm_matCL,Fnorm_matCR,Gamma,time)

    plot(log10.(Gamma*time),log10.(Fnorm_matCL),lw=4,label=L"L",palette=:reds,framestyle = :box)
    plot!(log10.(Gamma*time),log10.(Fnorm_matCR),lw=4,label=L"R",palette=:reds,framestyle = :box)

    # plot!(legend=:none)
    xlims!((-2.5,7))
    # ylims!((0,4))
    # plot!(size=(400,400))

end

function vNEfrommatC(val_matC::Union{Vector{Float64},Float64})

    vNE = 0.0

    for jj = 1:length(val_matC)
        if val_matC[jj] > 0 && val_matC[jj] < 1
           vNE += -val_matC[jj]*log(val_matC[jj]) - (1.0-val_matC[jj])*log(1.0-val_matC[jj])
        end
    end

    return vNE

end

function boundDrhopi(epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # correlation matrix
    K = length(epsilon)
    C0 = zeros(Float64,K)
    bound = 0

    for kk = 1:K
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
        if C0[kk] < 1/2
           bound += 1/C0[kk]
        elseif C0[kk] > 1/2
           bound += 1/(1 - C0[kk])
        # elseif
        end
    end

    return bound

end

function Esquare_bath(Ct::Union{Matrix{ComplexF64},Matrix{Float64}},epsilon::Vector{Float64})

    N = length(epsilon)
    Esquare = 0.0

    for ii = 1:N-1
        Esquare += epsilon[ii]^2*Ct[ii,ii]
        for jj = ii+1:N
            Esquare += epsilon[ii]*epsilon[jj]*(Ct[ii,ii]*Ct[jj,jj]-abs(Ct[ii,jj])^2)*2
        end
    end
    Esquare += epsilon[N]^2*Ct[N,N]

    return Esquare

end

function Nsquare_bath(Ct::Union{Matrix{ComplexF64},Matrix{Float64}})

    N = size(Ct)[1]
    Nsquare = 0.0

    for ii = 1:N-1
        Nsquare += Ct[ii,ii]
        for jj = ii+1:N
            Nsquare += (Ct[ii,ii]*Ct[jj,jj]-abs(Ct[ii,jj])^2)*2
        end
    end
    Nsquare += Ct[N,N]

    return Nsquare

end

function calculatequantities4(KL::Int64,KR::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian + fluctuated t
    matH = spzeros(Float64,KL+KR+1,KL+KR+1)
    createH_differentKLKR!(KL,KR,W,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = distribute_timepoint(Nt-1,0.0,tf)
    pushfirst!(time,0.0)

    # time = LinRange(0.0,tf,Nt)
    # dt = time[2] - time[1]
    # println("dt=",dt)
    # println("Note that int beta(t)*dQ/dt*dt depends on dt, so dt or tf/Nt should be small enough.")

    # correlation matrix
    # at initial
    C0 = zeros(Float64,KL+KR+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:KL
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
    end
    for kk = 1:KR
        C0[1+KL+kk] = 1.0/(exp((matH[1+KL+kk,1+KL+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # total enery and particle number, and estimated inverse temperature and chemical potential
    dC0 = diag(C0)
    E_tot0 = sum(dC0[1:KL+KR+1].*epsilonLR[1:KL+KR+1])
    N_tot0 = sum(dC0[1:KL+KR+1])
    Cgg0 = zeros(Float64,KL+KR+1)
    matCgg0 = zeros(Float64,KL+KR+1,KL+KR+1)
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,val_matH,vec_matH,invvec_matH)
    println("beta_gg=",effpara0[1])
    println("mu_gg=",effpara0[2])

    # global Gibbs state
    Cgg = globalGibbsstate(val_matH,vec_matH,invvec_matH,effpara0[1],effpara0[2])

    # mutual info between S and E
    val_Cgg = real(eigvals(Cgg))
    vNEgg = vNEfrommatC(val_Cgg)
    val_Cgg_sys = Cgg[1,1]
    vNEgg_sys = - val_Cgg_sys.*log.(val_Cgg_sys) - (1.0 .- val_Cgg_sys).*log.(1.0 .- val_Cgg_sys)
    val_Cgg_E = real(eigvals(Cgg[2:end,2:end]))
    vNEgg_E = vNEfrommatC(val_Cgg_E)
    Igg_SE = vNEgg_sys + vNEgg_E - vNEgg
    println("Igg_SE=",Igg_SE)

    # intrabath correlation
    val_Cgg_L = real(eigvals(Cgg[2:KL+1,2:KL+1]))
    vNEgg_L = vNEfrommatC(val_Cgg_L)
    val_Cgg_R = real(eigvals(Cgg[KL+2:end,KL+2:end]))
    vNEgg_R = vNEfrommatC(val_Cgg_R)
    Igg_B = vNEgg_L + vNEgg_R - vNEgg_E
    println("Igg_B=",Igg_B)

    # intramode correlation
    diag_Cgg_L = real(diag(Cgg[2:KL+1,2:KL+1]))
    vNEgg_Lk = vNEfrommatC(diag_Cgg_L)
    Igg_L = vNEgg_Lk - vNEgg_L
    diag_Cgg_R = real(diag(Cgg[KL+2:end,KL+2:end]))
    vNEgg_Rk = vNEfrommatC(diag_Cgg_R)
    Igg_R = vNEgg_Rk - vNEgg_R
    println("Igg_L=",Igg_L)
    println("Igg_R=",Igg_R)

    # variance
    EvarianceGibbs_global = Esquare_bath(Cgg,epsilonLR) - E_tot0^2
    NvarianceGibbs_global = Nsquare_bath(Cgg) - N_tot0^2
    println("EvarianceGibbs_global=",EvarianceGibbs_global)
    println("NvarianceGibbs_global=",NvarianceGibbs_global)
    println("EvarianceGibbs_global/E_tot0^2=",EvarianceGibbs_global/E_tot0^2)
    println("NvarianceGibbs_global/N_tot0^2=",NvarianceGibbs_global/N_tot0^2)

    # define space for input
    Ct = zeros(ComplexF64,KL+KR+1,KL+KR+1)
    dCt = zeros(ComplexF64,KL+KR+1)
    dCt1 = zeros(ComplexF64,KL+KR+1)
    val_Ct = zeros(Float64,KL+KR+1)
    val_Ct_E = zeros(Float64,KL+KR)
    diag_Ct_E = zeros(Float64,KL+KR)
    val_Ct_L = zeros(Float64,KL)
    val_Ct_R = zeros(Float64,KR)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    Evariance_L = zeros(ComplexF64,Nt)
    Evariance_R = zeros(ComplexF64,Nt)
    EvarianceGibbs_L = zeros(Float64,Nt)
    EvarianceGibbs_R = zeros(Float64,Nt)

    Nvariance_L = zeros(ComplexF64,Nt)
    Nvariance_R = zeros(ComplexF64,Nt)
    NvarianceGibbs_L = zeros(Float64,Nt)
    NvarianceGibbs_R = zeros(Float64,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    vNE_sys = zeros(Float64,Nt)
    vNE_E = zeros(Float64,Nt)
    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    vNE_alphak = zeros(Float64,Nt)
    vNE_Lk = zeros(Float64,Nt)
    vNE_Rk = zeros(Float64,Nt)
    vNE = zeros(Float64,Nt)

    I_SE = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)
    I_B = zeros(ComplexF64,Nt)
    I_L = zeros(ComplexF64,Nt)
    I_R = zeros(ComplexF64,Nt)

    QL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    dQLdt = zeros(ComplexF64,Nt)
    dQRdt = zeros(ComplexF64,Nt)
    betaQLtime = zeros(ComplexF64,Nt)
    betaQRtime = zeros(ComplexF64,Nt)

    Drel = zeros(ComplexF64,Nt)
    Drelnuk = zeros(ComplexF64,Nt)
    Drelpinuk = zeros(ComplexF64,Nt)

    sigma = zeros(ComplexF64,Nt)
    sigma2 = zeros(ComplexF64,Nt)
    sigma3 = zeros(ComplexF64,Nt)
    sigma_c = zeros(ComplexF64,Nt)

    CbathL = zeros(Float64,KL)
    CbathR = zeros(Float64,KR)
    matCL = zeros(Float64,2,2,Nt)
    matCR = zeros(Float64,2,2,Nt)

    deltavNEpiL = zeros(Float64,Nt)
    deltavNEpiR = zeros(Float64,Nt)
    sigma_c2 = zeros(ComplexF64,Nt)
    Drelpinuk2 = zeros(ComplexF64,Nt)

    boundL = zeros(Float64,Nt)
    boundR = zeros(Float64,Nt)

    deltavNEpiL0 = compute_vNEpi(epsilonLR[2:KL+1],betaL,muL)
    deltavNEpiR0 = compute_vNEpi(epsilonLR[KL+2:end],betaR,muR)

    Drel_rhoL_piL = zeros(Float64,Nt)
    Drel_rhoR_piR = zeros(Float64,Nt)
    Drel_rhoL_piL_ratio = zeros(Float64,Nt)
    Drel_rhoR_piR_ratio = zeros(Float64,Nt)

    # Threads.@threads for tt = 1:Nt
    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:KL+1].*epsilonLR[2:KL+1])
        E_R[tt] = sum(dCt[KL+2:end].*epsilonLR[KL+2:end])
        # E_k_L[:,tt] = dCt[2:K+1].*epsilonLR[2:K+1] # single site energy
        # E_k_R[:,tt] = dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1]

        Evariance_L[tt] = Esquare_bath(Ct[2:KL+1,2:KL+1],epsilonLR[2:KL+1]) - E_L[tt]^2
        Evariance_R[tt] = Esquare_bath(Ct[KL+2:end,KL+2:end],epsilonLR[KL+2:end]) - E_R[tt]^2

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2)

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:KL+1])
        N_R[tt] = sum(dCt[KL+2:end])
        # n_k_L[:,tt] = dCt[2:K+1] # single site occupation number
        # n_k_R[:,tt] = dCt[K+2:2*K+1]

        Nvariance_L[tt] = Nsquare_bath(Ct[2:KL+1,2:KL+1]) - N_L[tt]^2
        Nvariance_R[tt] = Nsquare_bath(Ct[KL+2:end,KL+2:end]) - N_R[tt]^2

        # vNE
        # total
        val_Ct .= real(eigvals(Ct))
        vNE[tt] = vNEfrommatC(val_Ct)
        # vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))
        # system
        vNE_sys[tt] = vNEfrommatC(real(Ct[1,1]))
        # vNE_sys[tt] = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        # environment
        val_Ct_E .= real(eigvals(Ct[2:end,2:end]))
        vNE_E[tt] = vNEfrommatC(val_Ct_E)
        # vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))

        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # mutual information describing the intraenvironment correlations
        diag_Ct_E .= real(diag(Ct[2:end,2:end]))
        vNE_alphak[tt] = vNEfrommatC(diag_Ct_E)
        # vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

        # I_B
        val_Ct_L .= real(eigvals(Ct[2:KL+1,2:KL+1]))
        vNE_L[tt] = vNEfrommatC(val_Ct_L)
        # vNE_L[tt] = - sum(val_Ct_L.*log.(val_Ct_L)) - sum((1.0 .- val_Ct_L).*log.(1.0 .- val_Ct_L))
        val_Ct_R .= real(eigvals(Ct[KL+2:end,KL+2:end]))
        vNE_R[tt] = vNEfrommatC(val_Ct_R)
        # vNE_R[tt] = - sum(val_Ct_R.*log.(val_Ct_R)) - sum((1.0 .- val_Ct_R).*log.(1.0 .- val_Ct_R))
        I_B[tt] = vNE_L[tt] + vNE_R[tt] - vNE_E[tt]

        # I_nu
        vNE_Lk[tt] = vNEfrommatC(diag_Ct_E[1:KL])
        # vNE_Lk[tt] = - sum(diag_Ct_E[1:KL].*log.(diag_Ct_E[1:KL])) - sum((1.0 .- diag_Ct_E[1:KL]).*log.(1.0 .- diag_Ct_E[1:KL]))
        I_L[tt] = vNE_Lk[tt] - vNE_L[tt]
        vNE_Rk[tt] = vNEfrommatC(diag_Ct_E[KL+1:end])
        # vNE_Rk[tt] = - sum(diag_Ct_E[KL+1:end].*log.(diag_Ct_E[KL+1:end])) - sum((1.0 .- diag_Ct_E[KL+1:end]).*log.(1.0 .- diag_Ct_E[KL+1:end]))
        I_R[tt] = vNE_Rk[tt] - vNE_R[tt]

        # effective inverse temperature and chemical potential
        betaL0 = betaL
        betaR0 = betaR
        muL0 = muL
        muR0 = muR
        if tt != 1
           betaL0 = effparaL[tt-1,1]
           betaR0 = effparaR[tt-1,1]
           muL0 = effparaL[tt-1,2]
           muR0 = effparaR[tt-1,2]
        end
        effparaL[tt,:] .= funeffectivebetamu(epsilonLR[2:KL+1],real(E_L[tt]),real(N_L[tt]),betaL0,muL0) #betaL,muL
        effparaR[tt,:] .= funeffectivebetamu(epsilonLR[KL+2:end],real(E_R[tt]),real(N_R[tt]),betaR0,muR0) #betaR,muR

        # heat
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:KL+1].*(epsilonLR[2:KL+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[KL+2:end].*(epsilonLR[KL+2:end] .- muR))
        betaQR[tt] = QR[tt]*betaR

        #
        # if tt != 1
        #    dQLdt[tt] = (QL[tt] - QL[tt-1])/dt
        #    dQRdt[tt] = (QR[tt] - QR[tt-1])/dt
        # end
        # betaQLtime[tt] = sum(dQLdt[1:tt].*effparaL[1:tt,1])*dt
        # betaQRtime[tt] = sum(dQRdt[1:tt].*effparaR[1:tt,1])*dt

        # heat capacity
        matCL[:,:,tt] = heatcapacityeff(CbathL,epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        matCR[:,:,tt] = heatcapacityeff(CbathR,epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])

        # variance of Gibbs states
        EvarianceGibbs_L[tt] = Evariance_Gibbs(CbathL,epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        EvarianceGibbs_R[tt] = Evariance_Gibbs(CbathR,epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])
        NvarianceGibbs_L[tt] = Nvariance_Gibbs(CbathL,epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        NvarianceGibbs_R[tt] = Nvariance_Gibbs(CbathR,epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])
        
        # relative entropy between rho_B(t) and rho_B(0)
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # relative entropy between rho_{nu,k}(t) and rho_{nu,k}(0)
        Drelnuk[tt] = Drel[tt] - I_env[tt]

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]
        sigma2[tt] = I_SE[tt] + Drel[tt]
        sigma3[tt] = I_SE[tt] + I_B[tt] + I_L[tt] + I_R[tt] + Drelnuk[tt]
        # sigma_c[tt] = vNE_sys[tt] - vNE_sys[1] - betaQLtime[tt] - betaQRtime[tt]

        # relative entropy between pi_nuk(t) and pi_nuk(0)
        Drelpinuk[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c[tt])  #sigma[tt] - sigma_c[tt]

        # vNE[beta(t)]-vNE[beta(0)] = int dt dQdt*beta(t)
        deltavNEpiL[tt] = compute_vNEpi(epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2]) - deltavNEpiL0
        deltavNEpiR[tt] = compute_vNEpi(epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2]) - deltavNEpiR0

        sigma_c2[tt] = vNE_sys[tt] - vNE_sys[1] + deltavNEpiL[tt] + deltavNEpiR[tt]
        Drelpinuk2[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c2[tt])

        # DrelrhopiL[tt] =  (deltavNEpiL[tt] + deltavNEpiL0) - vNE_L[tt] - I_B[tt]
        # DrelrhopiR[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c2[tt])

        boundL[tt] = boundDrhopi(epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        boundR[tt] = boundDrhopi(epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])

        # relative entropy between rho_nu(t) and pi_nu(t)
        Drel_rhoL_piL[tt] = -vNE_L[tt] + (deltavNEpiL[tt] + deltavNEpiL0)
        Drel_rhoR_piR[tt] = -vNE_R[tt] + (deltavNEpiR[tt] + deltavNEpiR0)
        # ratio with the bound
        Drel_rhoL_piL_ratio[tt] = Drel_rhoL_piL[tt]/(deltavNEpiL[tt] + deltavNEpiL0)
        Drel_rhoR_piR_ratio[tt] = Drel_rhoR_piR[tt]/(deltavNEpiR[tt] + deltavNEpiR0)

        println(tt)

    end

    # return time, vNE_sys, vNE_L, vNE_R, vNE

    return time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, Drelnuk, betaQL, betaQR, matCL, matCR, sigma_c2, Drelpinuk2, E_L, E_R, E_tot, N_L, N_R, boundL, boundR, Evariance_L, Evariance_R, EvarianceGibbs_L, EvarianceGibbs_R, Nvariance_L, Nvariance_R, NvarianceGibbs_L, NvarianceGibbs_R, Drel_rhoL_piL, Drel_rhoR_piR, Drel_rhoL_piL_ratio, Drel_rhoR_piR_ratio
    #E_k_L, E_k_R, n_k_L, n_k_R
    # return time, vNE_sys, effparaL, effparaR, QL, QR
    # return time, sigma, sigma3, sigma_c, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel
    # return time, sigma, sigma2, sigma3, sigma_c
    # return time, betaQL, betaQLtime, betaQR, betaQRtime
    # return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

function calculatequantities3(KL::Int64,KR::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian + fluctuated t
    matH = spzeros(Float64,KL+KR+1,KL+KR+1)
    createH_differentKLKR!(KL,KR,W,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = distribute_timepoint(Nt-1,0.0,tf)
    pushfirst!(time,0.0)

    # time = LinRange(0.0,tf,Nt)
    # dt = time[2] - time[1]
    # println("dt=",dt)
    # println("Note that int beta(t)*dQ/dt*dt depends on dt, so dt or tf/Nt should be small enough.")

    # correlation matrix
    # at initial
    C0 = zeros(Float64,KL+KR+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:KL
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
    end
    for kk = 1:KR
        C0[1+KL+kk] = 1.0/(exp((matH[1+KL+kk,1+KL+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # total enery and particle number, and estimated inverse temperature and chemical potential
    dC0 = diag(C0)
    E_tot0 = sum(dC0[1:KL+KR+1].*epsilonLR[1:KL+KR+1])
    N_tot0 = sum(dC0[1:KL+KR+1])
    Cgg0 = zeros(Float64,KL+KR+1)
    matCgg0 = zeros(Float64,KL+KR+1,KL+KR+1)
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,val_matH,vec_matH,invvec_matH)
    println("beta_gg=",effpara0[1])
    println("mu_gg=",effpara0[2])

    # global Gibbs state
    Cgg = globalGibbsstate(val_matH,vec_matH,invvec_matH,effpara0[1],effpara0[2])

    # mutual info between S and E
    val_Cgg = real(eigvals(Cgg))
    # vNEgg = vNEfrommatC(val_Cgg)
    vNEgg = - sum(val_Cgg.*log.(val_Cgg)) - sum((1.0 .- val_Cgg).*log.(1.0 .- val_Cgg))
    val_Cgg_sys = real(Cgg[1,1])
    vNEgg_sys = - val_Cgg_sys.*log.(val_Cgg_sys) - (1.0 .- val_Cgg_sys).*log.(1.0 .- val_Cgg_sys)
    val_Cgg_E = real(eigvals(Cgg[2:end,2:end]))
    # vNEgg_E = vNEfrommatC(val_Cgg_E)
    vNEgg_E = - sum(val_Cgg_E.*log.(val_Cgg_E)) - sum((1.0 .- val_Cgg_E).*log.(1.0 .- val_Cgg_E))
    Igg_SE = vNEgg_sys + vNEgg_E - vNEgg
    println("Igg_SE=",Igg_SE)

    # intrabath correlation
    val_Cgg_L = real(eigvals(Cgg[2:KL+1,2:KL+1]))
    # vNEgg_L = vNEfrommatC(val_Cgg_L)
    vNEgg_L = - sum(val_Cgg_L.*log.(val_Cgg_L)) - sum((1.0 .- val_Cgg_L).*log.(1.0 .- val_Cgg_L))
    val_Cgg_R = real(eigvals(Cgg[KL+2:end,KL+2:end]))
    # vNEgg_R = vNEfrommatC(val_Cgg_R)
    vNEgg_R = - sum(val_Cgg_R.*log.(val_Cgg_R)) - sum((1.0 .- val_Cgg_R).*log.(1.0 .- val_Cgg_R))
    Igg_B = vNEgg_L + vNEgg_R - vNEgg_E
    println("Igg_B=",Igg_B)

    # intramode correlation
    diag_Cgg_L = real(diag(Cgg[2:KL+1,2:KL+1]))
    # vNEgg_Lk = vNEfrommatC(diag_Cgg_L)
    vNEgg_Lk = - sum(diag_Cgg_L.*log.(diag_Cgg_L)) - sum((1.0 .- diag_Cgg_L).*log.(1.0 .- diag_Cgg_L))
    Igg_L = vNEgg_Lk - vNEgg_L
    diag_Cgg_R = real(diag(Cgg[KL+2:end,KL+2:end]))
    # vNEgg_Rk = vNEfrommatC(diag_Cgg_R)
    vNEgg_Rk = - sum(diag_Cgg_R.*log.(diag_Cgg_R)) - sum((1.0 .- diag_Cgg_R).*log.(1.0 .- diag_Cgg_R))
    Igg_R = vNEgg_Rk - vNEgg_R
    println("Igg_L=",Igg_L)
    println("Igg_R=",Igg_R)

    # define space for input
    Ct = zeros(ComplexF64,KL+KR+1,KL+KR+1)
    dCt = zeros(ComplexF64,KL+KR+1)
    dCt1 = zeros(ComplexF64,KL+KR+1)
    val_Ct = zeros(Float64,KL+KR+1)
    val_Ct_E = zeros(Float64,KL+KR)
    diag_Ct_E = zeros(Float64,KL+KR)
    val_Ct_L = zeros(Float64,KL)
    val_Ct_R = zeros(Float64,KR)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    vNE_sys = zeros(Float64,Nt)
    vNE_E = zeros(Float64,Nt)
    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    vNE_alphak = zeros(Float64,Nt)
    vNE_Lk = zeros(Float64,Nt)
    vNE_Rk = zeros(Float64,Nt)
    vNE = zeros(Float64,Nt)

    I_SE = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)
    I_B = zeros(ComplexF64,Nt)
    I_L = zeros(ComplexF64,Nt)
    I_R = zeros(ComplexF64,Nt)

    QL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    dQLdt = zeros(ComplexF64,Nt)
    dQRdt = zeros(ComplexF64,Nt)
    betaQLtime = zeros(ComplexF64,Nt)
    betaQRtime = zeros(ComplexF64,Nt)

    Drel = zeros(ComplexF64,Nt)
    Drelnuk = zeros(ComplexF64,Nt)
    Drelpinuk = zeros(ComplexF64,Nt)

    sigma = zeros(ComplexF64,Nt)
    sigma2 = zeros(ComplexF64,Nt)
    sigma3 = zeros(ComplexF64,Nt)
    sigma_c = zeros(ComplexF64,Nt)

    CbathL = zeros(Float64,KL)
    CbathR = zeros(Float64,KR)
    matCL = zeros(Float64,2,2,Nt)
    matCR = zeros(Float64,2,2,Nt)

    deltavNEpiL = zeros(Float64,Nt)
    deltavNEpiR = zeros(Float64,Nt)
    sigma_c2 = zeros(ComplexF64,Nt)
    Drelpinuk2 = zeros(ComplexF64,Nt)

    deltavNEpiL0 = compute_vNEpi(epsilonLR[2:KL+1],betaL,muL)
    deltavNEpiR0 = compute_vNEpi(epsilonLR[KL+2:end],betaR,muR)

    # Threads.@threads for tt = 1:Nt
    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:KL+1].*epsilonLR[2:KL+1])
        E_R[tt] = sum(dCt[KL+2:end].*epsilonLR[KL+2:end])
        # E_k_L[:,tt] = dCt[2:K+1].*epsilonLR[2:K+1] # single site energy
        # E_k_R[:,tt] = dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1]

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2)

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:KL+1])
        N_R[tt] = sum(dCt[KL+2:end])
        # n_k_L[:,tt] = dCt[2:K+1] # single site occupation number
        # n_k_R[:,tt] = dCt[K+2:2*K+1]

        # vNE
        # total
        val_Ct .= real(eigvals(Ct))
        # vNE[tt] = vNEfrommatC(val_Ct)
        vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))
        # system
        val_Ct_sys = real(Ct[1,1])
        # vNE_sys[tt] = vNEfrommatC(real(Ct[1,1]))
        vNE_sys[tt] = -val_Ct_sys*log(val_Ct_sys) - (1-val_Ct_sys)*log(1-val_Ct_sys)
        # environment
        val_Ct_E .= real(eigvals(Ct[2:end,2:end]))
        # vNE_E[tt] = vNEfrommatC(val_Ct_E)
        vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))


        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # mutual information describing the intraenvironment correlations
        diag_Ct_E .= real(diag(Ct[2:end,2:end]))
        # vNE_alphak[tt] = vNEfrommatC(diag_Ct_E)
        vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

        # I_B
        val_Ct_L .= real(eigvals(Ct[2:KL+1,2:KL+1]))
        # vNE_L[tt] = vNEfrommatC(val_Ct_L)
        vNE_L[tt] = - sum(val_Ct_L.*log.(val_Ct_L)) - sum((1.0 .- val_Ct_L).*log.(1.0 .- val_Ct_L))
        val_Ct_R .= real(eigvals(Ct[KL+2:end,KL+2:end]))
        # vNE_R[tt] = vNEfrommatC(val_Ct_R)
        vNE_R[tt] = - sum(val_Ct_R.*log.(val_Ct_R)) - sum((1.0 .- val_Ct_R).*log.(1.0 .- val_Ct_R))
        I_B[tt] = vNE_L[tt] + vNE_R[tt] - vNE_E[tt]

        # I_nu
        # vNE_Lk[tt] = vNEfrommatC(diag_Ct_E[1:KL])
        vNE_Lk[tt] = - sum(diag_Ct_E[1:KL].*log.(diag_Ct_E[1:KL])) - sum((1.0 .- diag_Ct_E[1:KL]).*log.(1.0 .- diag_Ct_E[1:KL]))
        I_L[tt] = vNE_Lk[tt] - vNE_L[tt]
        # vNE_Rk[tt] = vNEfrommatC(diag_Ct_E[KL+1:end])
        vNE_Rk[tt] = - sum(diag_Ct_E[KL+1:end].*log.(diag_Ct_E[KL+1:end])) - sum((1.0 .- diag_Ct_E[KL+1:end]).*log.(1.0 .- diag_Ct_E[KL+1:end]))
        I_R[tt] = vNE_Rk[tt] - vNE_R[tt]

        # effective inverse temperature and chemical potential
        betaL0 = betaL
        betaR0 = betaR
        muL0 = muL
        muR0 = muR
        if tt != 1
           betaL0 = effparaL[tt-1,1]
           betaR0 = effparaR[tt-1,1]
           muL0 = effparaL[tt-1,2]
           muR0 = effparaR[tt-1,2]
        end
        effparaL[tt,:] .= funeffectivebetamu(epsilonLR[2:KL+1],real(E_L[tt]),real(N_L[tt]),betaL0,muL0) #betaL,muL
        effparaR[tt,:] .= funeffectivebetamu(epsilonLR[KL+2:end],real(E_R[tt]),real(N_R[tt]),betaR0,muR0) #betaR,muR

        # heat
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:KL+1].*(epsilonLR[2:KL+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[KL+2:end].*(epsilonLR[KL+2:end] .- muR))
        betaQR[tt] = QR[tt]*betaR

        #
        # if tt != 1
        #    dQLdt[tt] = (QL[tt] - QL[tt-1])/dt
        #    dQRdt[tt] = (QR[tt] - QR[tt-1])/dt
        # end
        # betaQLtime[tt] = sum(dQLdt[1:tt].*effparaL[1:tt,1])*dt
        # betaQRtime[tt] = sum(dQRdt[1:tt].*effparaR[1:tt,1])*dt

        # heat capacity
        matCL[:,:,tt] = heatcapacityeff(CbathL,epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        matCR[:,:,tt] = heatcapacityeff(CbathR,epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])

        # relative entropy between rho_B(t) and rho_B(0)
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # relative entropy between rho_{nu,k}(t) and rho_{nu,k}(0)
        Drelnuk[tt] = Drel[tt] - I_env[tt]

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]
        sigma2[tt] = I_SE[tt] + Drel[tt]
        sigma3[tt] = I_SE[tt] + I_B[tt] + I_L[tt] + I_R[tt] + Drelnuk[tt]
        # sigma_c[tt] = vNE_sys[tt] - vNE_sys[1] - betaQLtime[tt] - betaQRtime[tt]

        # relative entropy between pi_nuk(t) and pi_nuk(0)
        Drelpinuk[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c[tt])  #sigma[tt] - sigma_c[tt]

        # vNE[beta(t)]-vNE[beta(0)] = int dt dQdt*beta(t)
        deltavNEpiL[tt] = compute_vNEpi(epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2]) - deltavNEpiL0
        deltavNEpiR[tt] = compute_vNEpi(epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2]) - deltavNEpiR0

        sigma_c2[tt] = vNE_sys[tt] - vNE_sys[1] + deltavNEpiL[tt] + deltavNEpiR[tt]
        Drelpinuk2[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c2[tt])

        println(tt)

    end

    # return time, vNE_sys, vNE_L, vNE_R, vNE

    return time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, Drelnuk, betaQL, betaQR, matCL, matCR, sigma_c2, Drelpinuk2, E_L, E_R, E_tot
    #E_k_L, E_k_R, n_k_L, n_k_R
    # return time, vNE_sys, effparaL, effparaR, QL, QR
    # return time, sigma, sigma3, sigma_c, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel
    # return time, sigma, sigma2, sigma3, sigma_c
    # return time, betaQL, betaQLtime, betaQR, betaQRtime
    # return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

function calculatequantities2(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian + fluctuated t
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_fluctuatedt!(K,W,t_flu,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)
    dt = time[2] - time[1]
    # println("dt=",dt)
    # println("Note that int beta(t)*dQ/dt*dt depends on dt, so dt or tf/Nt should be small enough.")

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # total enery and particle number, and estimated inverse temperature and chemical potential
    dC0 = diag(C0)
    E_tot0 = sum(dC0[1:2*K+1].*epsilonLR[1:2*K+1])
    N_tot0 = sum(dC0[1:2*K+1])
    # effpara0 = funeffectivebetamu(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2)
    Cgg0 = zeros(Float64,K*2+1)
    matCgg0 = zeros(Float64,K*2+1,K*2+1)
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,val_matH,vec_matH,invvec_matH)
    println("beta_gg=",effpara0[1])
    println("mu_gg=",effpara0[2])

    # global Gibbs state
    Cgg = globalGibbsstate(val_matH,vec_matH,invvec_matH,effpara0[1],effpara0[2])

    # mutual info between S and E
    val_Cgg = eigvals(Cgg)
    vNEgg = - sum(val_Cgg.*log.(val_Cgg)) - sum((1.0 .- val_Cgg).*log.(1.0 .- val_Cgg))
    val_Cgg_sys = Cgg[1,1]
    vNEgg_sys = - val_Cgg_sys.*log.(val_Cgg_sys) - (1.0 .- val_Cgg_sys).*log.(1.0 .- val_Cgg_sys)
    val_Cgg_E = eigvals(Cgg[2:2*K+1,2:2*K+1])
    vNEgg_E = - sum(val_Cgg_E.*log.(val_Cgg_E)) - sum((1.0 .- val_Cgg_E).*log.(1.0 .- val_Cgg_E))
    Igg_SE = vNEgg_sys + vNEgg_E - vNEgg
    println("Igg_SE=",Igg_SE)

    # intrabath correlation
    val_Cgg_L = eigvals(Cgg[2:K+1,2:K+1])
    vNEgg_L = - sum(val_Cgg_L.*log.(val_Cgg_L)) - sum((1.0 .- val_Cgg_L).*log.(1.0 .- val_Cgg_L))
    val_Cgg_R = eigvals(Cgg[K+2:2*K+1,K+2:2*K+1])
    vNEgg_R = - sum(val_Cgg_R.*log.(val_Cgg_R)) - sum((1.0 .- val_Cgg_R).*log.(1.0 .- val_Cgg_R))
    Igg_B = vNEgg_L + vNEgg_R - vNEgg_E
    println("Igg_B=",Igg_B)

    # intramode correlation
    diag_Cgg_E = diag(Cgg[2:end,2:end])
    vNEgg_Lk = - sum(diag_Cgg_E[1:K].*log.(diag_Cgg_E[1:K])) - sum((1.0 .- diag_Cgg_E[1:K]).*log.(1.0 .- diag_Cgg_E[1:K]))
    Igg_L = vNEgg_Lk - vNEgg_L
    vNEgg_Rk = - sum(diag_Cgg_E[K+1:2*K].*log.(diag_Cgg_E[K+1:2*K])) - sum((1.0 .- diag_Cgg_E[K+1:2*K]).*log.(1.0 .- diag_Cgg_E[K+1:2*K]))
    Igg_R = vNEgg_Rk - vNEgg_R
    println("Igg_L=",Igg_L)
    println("Igg_R=",Igg_R)

    # define space for input
    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    dCt = zeros(ComplexF64,K*2+1)
    dCt1 = zeros(ComplexF64,K*2+1)
    val_Ct = zeros(ComplexF64,K*2+1)
    val_Ct_E = zeros(ComplexF64,K*2)
    diag_Ct_E = zeros(ComplexF64,K*2)
    val_Ct_L = zeros(ComplexF64,K)
    val_Ct_R = zeros(ComplexF64,K)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_L = zeros(ComplexF64,Nt)
    vNE_R = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    vNE_Lk = zeros(ComplexF64,Nt)
    vNE_Rk = zeros(ComplexF64,Nt)
    vNE = zeros(ComplexF64,Nt)

    I_SE = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)
    I_B = zeros(ComplexF64,Nt)
    I_L = zeros(ComplexF64,Nt)
    I_R = zeros(ComplexF64,Nt)

    QL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    dQLdt = zeros(ComplexF64,Nt)
    dQRdt = zeros(ComplexF64,Nt)
    betaQLtime = zeros(ComplexF64,Nt)
    betaQRtime = zeros(ComplexF64,Nt)

    Drel = zeros(ComplexF64,Nt)
    Drelnuk = zeros(ComplexF64,Nt)
    Drelpinuk = zeros(ComplexF64,Nt)

    sigma = zeros(ComplexF64,Nt)
    sigma2 = zeros(ComplexF64,Nt)
    sigma3 = zeros(ComplexF64,Nt)
    sigma_c = zeros(ComplexF64,Nt)

    Cbath = zeros(Float64,K)
    matCL = zeros(Float64,2,2,Nt)
    matCR = zeros(Float64,2,2,Nt)

    deltavNEpiL = zeros(Float64,Nt)
    deltavNEpiR = zeros(Float64,Nt)
    sigma_c2 = zeros(ComplexF64,Nt)
    Drelpinuk2 = zeros(ComplexF64,Nt)

    deltavNEpiL0 = compute_vNEpi(epsilonLR[2:K+1],betaL,muL)
    deltavNEpiR0 = compute_vNEpi(epsilonLR[K+2:2*K+1],betaR,muR)

    # Threads.@threads for tt = 1:Nt
    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:K+1].*epsilonLR[2:K+1])
        E_R[tt] = sum(dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1])
        # E_k_L[:,tt] = dCt[2:K+1].*epsilonLR[2:K+1] # single site energy
        # E_k_R[:,tt] = dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1]

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2)

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:K+1])
        N_R[tt] = sum(dCt[K+2:2*K+1])
        # n_k_L[:,tt] = dCt[2:K+1] # single site occupation number
        # n_k_R[:,tt] = dCt[K+2:2*K+1]

        # vNE
        # total
        val_Ct .= eigvals(Ct)
        vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))
        # system
        vNE_sys[tt] = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        # environment
        val_Ct_E .= eigvals(Ct[2:end,2:end])
        vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))

        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # mutual information describing the intraenvironment correlations
        diag_Ct_E .= diag(Ct[2:end,2:end])
        vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

        # I_B
        val_Ct_L .= eigvals(Ct[2:K+1,2:K+1])
        vNE_L[tt] = - sum(val_Ct_L.*log.(val_Ct_L)) - sum((1.0 .- val_Ct_L).*log.(1.0 .- val_Ct_L))
        val_Ct_R .= eigvals(Ct[K+2:2*K+1,K+2:2*K+1])
        vNE_R[tt] = - sum(val_Ct_R.*log.(val_Ct_R)) - sum((1.0 .- val_Ct_R).*log.(1.0 .- val_Ct_R))
        I_B[tt] = vNE_L[tt] + vNE_R[tt] - vNE_E[tt]

        # I_nu
        vNE_Lk[tt] = - sum(diag_Ct_E[1:K].*log.(diag_Ct_E[1:K])) - sum((1.0 .- diag_Ct_E[1:K]).*log.(1.0 .- diag_Ct_E[1:K]))
        I_L[tt] = vNE_Lk[tt] - vNE_L[tt]
        vNE_Rk[tt] = - sum(diag_Ct_E[K+1:2*K].*log.(diag_Ct_E[K+1:2*K])) - sum((1.0 .- diag_Ct_E[K+1:2*K]).*log.(1.0 .- diag_Ct_E[K+1:2*K]))
        I_R[tt] = vNE_Rk[tt] - vNE_R[tt]

        # effective inverse temperature and chemical potential
        betaL0 = betaL
        betaR0 = betaR
        muL0 = muL
        muR0 = muR
        if tt != 1
           betaL0 = effparaL[tt-1,1]
           betaR0 = effparaR[tt-1,1]
           muL0 = effparaL[tt-1,2]
           muR0 = effparaR[tt-1,2]
        end
        effparaL[tt,:] .= funeffectivebetamu(epsilonLR[2:K+1],real(E_L[tt]),real(N_L[tt]),betaL0,muL0) #betaL,muL
        effparaR[tt,:] .= funeffectivebetamu(epsilonLR[K+2:2*K+1],real(E_R[tt]),real(N_R[tt]),betaR0,muR0) #betaR,muR

        # heat
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        betaQR[tt] = QR[tt]*betaR

        #
        if tt != 1
           dQLdt[tt] = (QL[tt] - QL[tt-1])/dt
           dQRdt[tt] = (QR[tt] - QR[tt-1])/dt
        end
        betaQLtime[tt] = sum(dQLdt[1:tt].*effparaL[1:tt,1])*dt
        betaQRtime[tt] = sum(dQRdt[1:tt].*effparaR[1:tt,1])*dt

        # heat capacity
        matCL[:,:,tt] = heatcapacityeff(Cbath,epsilonLR[2:K+1],effparaL[tt,1],effparaL[tt,2])
        matCR[:,:,tt] = heatcapacityeff(Cbath,epsilonLR[K+2:2*K+1],effparaR[tt,1],effparaR[tt,2])

        # relative entropy between rho_B(t) and rho_B(0)
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # relative entropy between rho_{nu,k}(t) and rho_{nu,k}(0)
        Drelnuk[tt] = Drel[tt] - I_env[tt]

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]
        sigma2[tt] = I_SE[tt] + Drel[tt]
        sigma3[tt] = I_SE[tt] + I_B[tt] + I_L[tt] + I_R[tt] + Drelnuk[tt]
        sigma_c[tt] = vNE_sys[tt] - vNE_sys[1] - betaQLtime[tt] - betaQRtime[tt]

        # relative entropy between pi_nuk(t) and pi_nuk(0)
        Drelpinuk[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c[tt])  #sigma[tt] - sigma_c[tt]

        # vNE[beta(t)]-vNE[beta(0)] = int dt dQdt*beta(t)
        deltavNEpiL[tt] = compute_vNEpi(epsilonLR[2:K+1],effparaL[tt,1],effparaL[tt,2]) - deltavNEpiL0
        deltavNEpiR[tt] = compute_vNEpi(epsilonLR[K+2:2*K+1],effparaR[tt,1],effparaR[tt,2]) - deltavNEpiR0

        sigma_c2[tt] = vNE_sys[tt] - vNE_sys[1] + deltavNEpiL[tt] + deltavNEpiR[tt]
        Drelpinuk2[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c2[tt])

    end

    # return time, vNE_sys, vNE_L, vNE_R, vNE

    return time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR, deltavNEpiL, deltavNEpiR, sigma_c2, Drelpinuk2
    #E_k_L, E_k_R, n_k_L, n_k_R
    # return time, vNE_sys, effparaL, effparaR, QL, QR
    # return time, sigma, sigma3, sigma_c, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel
    # return time, sigma, sigma2, sigma3, sigma_c
    # return time, betaQL, betaQLtime, betaQR, betaQRtime
    # return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

# save("data_calculatequantities2_K128W20betaL1R05GammaL05R05muL1R1tf1000Nt10001.jld", "time", time, "sigma", sigma, "sigma3", sigma3, "sigma_c", sigma_c, "effparaL", effparaL, "effparaR", effparaR, "I_SE", I_SE, "I_B", I_B, "I_L", I_L, "I_R", I_R, "I_env", I_env, "Drel", Drel)

function plot_Fnorm_C(time,matCL,matCR)

    Nt = length(time)
    Fnorm_matCL = zeros(Float64,Nt)
    Fnorm_matCR = zeros(Float64,Nt)

    for tt = 1:Nt
        Fnorm_matCL[tt] = sqrt(tr(matCL[:,:,tt]*matCL[:,:,tt]'))
        Fnorm_matCR[tt] = sqrt(tr(matCR[:,:,tt]*matCR[:,:,tt]'))
    end

    return Fnorm_matCL, Fnorm_matCR

    # plot(log10.(time),log10.(Fnorm_matCL))
    # plot!(log10.(time),log10.(Fnorm_matCR))
    # plot!(legend=:none)
    # plot!(xlabel=L"\log_{10}\Gamma t")
    # ylims!((0,3.1))

end

function calculatequantities2_singlefermions(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian + fluctuated t
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_fluctuatedt!(K,W,t_flu,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)
    dt = time[2] - time[1]
    println("dt=",dt)
    println("Note that int beta(t)*dQ/dt*dt depends on dt, so dt or tf/Nt should be small enough.")

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # total enery and particle number, and estimated inverse temperature and chemical potential
    dC0 = diag(C0)
    E_tot0 = sum(dC0[1:2*K+1].*epsilonLR[1:2*K+1])
    N_tot0 = sum(dC0[1:2*K+1])
    # effpara0 = funeffectivebetamu(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2)
    Cgg0 = zeros(Float64,K*2+1)
    matCgg0 = zeros(Float64,K*2+1,K*2+1)
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,K,K,val_matH,vec_matH,invvec_matH)
    println("beta_gg=",effpara0[1])
    println("mu_gg=",effpara0[2])

    # global Gibbs state
    Cgg = globalGibbsstate(K,val_matH,vec_matH,invvec_matH,effpara0[1],effpara0[2])

    # mutual info between S and E
    val_Cgg = eigvals(Cgg)
    vNEgg = - sum(val_Cgg.*log.(val_Cgg)) - sum((1.0 .- val_Cgg).*log.(1.0 .- val_Cgg))
    val_Cgg_sys = Cgg[1,1]
    vNEgg_sys = - val_Cgg_sys.*log.(val_Cgg_sys) - (1.0 .- val_Cgg_sys).*log.(1.0 .- val_Cgg_sys)
    val_Cgg_E = eigvals(Cgg[2:2*K+1,2:2*K+1])
    vNEgg_E = - sum(val_Cgg_E.*log.(val_Cgg_E)) - sum((1.0 .- val_Cgg_E).*log.(1.0 .- val_Cgg_E))
    Igg_SE = vNEgg_sys + vNEgg_E - vNEgg
    println("Igg_SE=",Igg_SE)

    # intrabath correlation
    val_Cgg_L = eigvals(Cgg[2:K+1,2:K+1])
    vNEgg_L = - sum(val_Cgg_L.*log.(val_Cgg_L)) - sum((1.0 .- val_Cgg_L).*log.(1.0 .- val_Cgg_L))
    val_Cgg_R = eigvals(Cgg[K+2:2*K+1,K+2:2*K+1])
    vNEgg_R = - sum(val_Cgg_R.*log.(val_Cgg_R)) - sum((1.0 .- val_Cgg_R).*log.(1.0 .- val_Cgg_R))
    Igg_B = vNEgg_L + vNEgg_R - vNEgg_E
    println("Igg_B=",Igg_B)

    # intramode correlation
    diag_Cgg_E = diag(Cgg[2:end,2:end])
    vNEgg_Lk = - sum(diag_Cgg_E[1:K].*log.(diag_Cgg_E[1:K])) - sum((1.0 .- diag_Cgg_E[1:K]).*log.(1.0 .- diag_Cgg_E[1:K]))
    Igg_L = vNEgg_Lk - vNEgg_L
    vNEgg_Rk = - sum(diag_Cgg_E[K+1:2*K].*log.(diag_Cgg_E[K+1:2*K])) - sum((1.0 .- diag_Cgg_E[K+1:2*K]).*log.(1.0 .- diag_Cgg_E[K+1:2*K]))
    Igg_R = vNEgg_Rk - vNEgg_R
    println("Igg_L=",Igg_L)
    println("Igg_R=",Igg_R)

    # define space for input
    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    dCt = zeros(ComplexF64,K*2+1)
    dCt1 = zeros(ComplexF64,K*2+1)
    val_Ct = zeros(ComplexF64,K*2+1)
    val_Ct_E = zeros(ComplexF64,K*2)
    diag_Ct_E = zeros(ComplexF64,K*2)
    val_Ct_L = zeros(ComplexF64,K)
    val_Ct_R = zeros(ComplexF64,K)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    E_k_L = zeros(ComplexF64,K,Nt)
    E_k_R = zeros(ComplexF64,K,Nt)
    n_k_L = zeros(ComplexF64,K,Nt)
    n_k_R = zeros(ComplexF64,K,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_L = zeros(ComplexF64,Nt)
    vNE_R = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    vNE_Lk = zeros(ComplexF64,Nt)
    vNE_Rk = zeros(ComplexF64,Nt)
    vNE = zeros(ComplexF64,Nt)

    I_SE = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)
    I_B = zeros(ComplexF64,Nt)
    I_L = zeros(ComplexF64,Nt)
    I_R = zeros(ComplexF64,Nt)

    QL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    dQLdt = zeros(ComplexF64,Nt)
    dQRdt = zeros(ComplexF64,Nt)
    betaQLtime = zeros(ComplexF64,Nt)
    betaQRtime = zeros(ComplexF64,Nt)

    Drel = zeros(ComplexF64,Nt)
    Drelnuk = zeros(ComplexF64,Nt)
    Drelpinuk = zeros(ComplexF64,Nt)

    sigma = zeros(ComplexF64,Nt)
    sigma2 = zeros(ComplexF64,Nt)
    sigma3 = zeros(ComplexF64,Nt)
    sigma_c = zeros(ComplexF64,Nt)

    Cbath = zeros(Float64,K)
    matCL = zeros(Float64,2,2,Nt)
    matCR = zeros(Float64,2,2,Nt)

    # Threads.@threads for tt = 1:Nt
    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:K+1].*epsilonLR[2:K+1])
        E_R[tt] = sum(dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1])
        E_k_L[:,tt] = dCt[2:K+1].*epsilonLR[2:K+1] # single site energy
        E_k_R[:,tt] = dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1]

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2)

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:K+1])
        N_R[tt] = sum(dCt[K+2:2*K+1])
        n_k_L[:,tt] = dCt[2:K+1] # single site occupation number
        n_k_R[:,tt] = dCt[K+2:2*K+1]

        # vNE
        # total
        val_Ct .= eigvals(Ct)
        vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))
        # system
        vNE_sys[tt] = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        # environment
        val_Ct_E .= eigvals(Ct[2:end,2:end])
        vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))

        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # mutual information describing the intraenvironment correlations
        diag_Ct_E .= diag(Ct[2:end,2:end])
        vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

        # I_B
        val_Ct_L .= eigvals(Ct[2:K+1,2:K+1])
        vNE_L[tt] = - sum(val_Ct_L.*log.(val_Ct_L)) - sum((1.0 .- val_Ct_L).*log.(1.0 .- val_Ct_L))
        val_Ct_R .= eigvals(Ct[K+2:2*K+1,K+2:2*K+1])
        vNE_R[tt] = - sum(val_Ct_R.*log.(val_Ct_R)) - sum((1.0 .- val_Ct_R).*log.(1.0 .- val_Ct_R))
        I_B[tt] = vNE_L[tt] + vNE_R[tt] - vNE_E[tt]

        # I_nu
        vNE_Lk[tt] = - sum(diag_Ct_E[1:K].*log.(diag_Ct_E[1:K])) - sum((1.0 .- diag_Ct_E[1:K]).*log.(1.0 .- diag_Ct_E[1:K]))
        I_L[tt] = vNE_Lk[tt] - vNE_L[tt]
        vNE_Rk[tt] = - sum(diag_Ct_E[K+1:2*K].*log.(diag_Ct_E[K+1:2*K])) - sum((1.0 .- diag_Ct_E[K+1:2*K]).*log.(1.0 .- diag_Ct_E[K+1:2*K]))
        I_R[tt] = vNE_Rk[tt] - vNE_R[tt]

        # effective inverse temperature and chemical potential
        betaL0 = betaL
        betaR0 = betaR
        muL0 = muL
        muR0 = muR
        if tt != 1
           betaL0 = effparaL[tt-1,1]
           betaR0 = effparaR[tt-1,1]
           muL0 = effparaL[tt-1,2]
           muR0 = effparaR[tt-1,2]
        end
        effparaL[tt,:] .= funeffectivebetamu(epsilonLR[2:K+1],real(E_L[tt]),real(N_L[tt]),betaL0,muL0) #betaL,muL
        effparaR[tt,:] .= funeffectivebetamu(epsilonLR[K+2:2*K+1],real(E_R[tt]),real(N_R[tt]),betaR0,muR0) #betaR,muR

        # heat
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        betaQR[tt] = QR[tt]*betaR

        #
        if tt != 1
           dQLdt[tt] = (QL[tt] - QL[tt-1])/dt
           dQRdt[tt] = (QR[tt] - QR[tt-1])/dt
        end
        betaQLtime[tt] = sum(dQLdt[1:tt].*effparaL[1:tt,1])*dt
        betaQRtime[tt] = sum(dQRdt[1:tt].*effparaR[1:tt,1])*dt

        # heat capacity
        matCL[:,:,tt] = heatcapacityeff(Cbath,K,epsilonLR[2:K+1],effparaL[tt,1],effparaL[tt,2])
        matCR[:,:,tt] = heatcapacityeff(Cbath,K,epsilonLR[K+2:2*K+1],effparaR[tt,1],effparaR[tt,2])

        # relative entropy between rho_B(t) and rho_B(0)
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # relative entropy between rho_{nu,k}(t) and rho_{nu,k}(0)
        Drelnuk[tt] = Drel[tt] - I_env[tt]

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]
        sigma2[tt] = I_SE[tt] + Drel[tt]
        sigma3[tt] = I_SE[tt] + I_B[tt] + I_L[tt] + I_R[tt] + Drelnuk[tt]
        sigma_c[tt] = vNE_sys[tt] - vNE_sys[1] - betaQLtime[tt] - betaQRtime[tt]

        # relative entropy between pi_nuk(t) and pi_nuk(0)
        Drelpinuk[tt] =  Drelnuk[tt] - (sigma[tt] - sigma_c[tt])  #sigma[tt] - sigma_c[tt]

    end

    # return time, vNE_sys, vNE_L, vNE_R, vNE

    return time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR, E_L, E_R, N_L, N_R, E_k_L, E_k_R, n_k_L, n_k_R
    # return time, vNE_sys, effparaL, effparaR, QL, QR
    # return time, sigma, sigma3, sigma_c, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel
    # return time, sigma, sigma2, sigma3, sigma_c
    # return time, betaQL, betaQLtime, betaQR, betaQRtime
    # return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

function plot_E_k_LR(K,Gamma,time,E_k_L,E_L)

    p1 = plot(log10.(time*Gamma),real(E_k_L[1,:]))

    for jj = 2:K
        plot!(log10.(time*Gamma),real(E_k_L[jj,:]))
    end

    p2 = plot(log10.(time*Gamma),real(E_L))

    plot(p1,p2,layout=(2,1)) #,size=(800,300),dpi=600)

end

function calculatequantities(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # reproduce the results of Phys. Rev. Lett. 123, 200603 (2019)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # Depsilon = W/(K-1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    dCt = zeros(ComplexF64,K*2+1)
    Ct_E = zeros(ComplexF64,K*2,K*2)
    val_Ct_E = zeros(ComplexF64,K*2)
    epsilonLR = zeros(ComplexF64,K*2+1)

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    I_SE = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    Drel = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)

    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH
        # Ct = Hermitian(Ct)

        # vNE
        vNE_sys[tt] = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        Ct_E .= Ct[2:end,2:end]
        val_Ct_E .= eigvals(Ct_E)
        vNE_E[tt] = - sum(val_Ct_E.*log.(val_Ct_E)) - sum((1.0 .- val_Ct_E).*log.(1.0 .- val_Ct_E))
        # println(vNE_sys[tt])

        # I_SE
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # heat
        dCt .= diag(Ct - C0)
        epsilonLR .= diag(matH)
        betaQL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))*betaL
        betaQR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))*betaR

        # entropy production
        sigma[tt] = vNE_sys[tt] - vNE_sys[1] - betaQL[tt] - betaQR[tt]

        # relative entropy
        Drel[tt] = - betaQL[tt] - betaQR[tt] - (vNE_E[tt] - vNE_E[1])

        # mutual information describing the intraenvironment correlations
        diag_Ct_E = diag(Ct_E)
        vNE_alphak[tt] = - sum(diag_Ct_E.*log.(diag_Ct_E)) - sum((1.0 .- diag_Ct_E).*log.(1.0 .- diag_Ct_E))
        I_env[tt] = vNE_alphak[tt] - vNE_E[tt]

    end

    return time, vNE_E.-vNE_E[1], I_SE, betaQL, betaQR, sigma, Drel, I_env

end

function movingmean(x::Vector{Float64}, n::Int64)

    if iseven(n)
       error("use an odd number for n to make the resulting moving average symmetric.")
    end

    m = zeros(Float64,length(x))
    δ = Int64((n-1)/2) #ceil(Int64,(n-1)/2)
    for ii = 1:length(x)
        ii0 = max(ii-δ,1)
        ii1 = min(ii+δ,length(x))
        m[ii] = mean(x[ii0:ii1])
    end

    return m

end

function plot_sigmas_old(time,GammaLR,sigma,sigma_c,I_SE,I_B,I_L,I_R,Drelnuk,Drelpinuk)

    I_SE_mvave = movingmean(real(I_SE),1001);
    I_L_mvave = movingmean(real(I_L),1001);
    I_R_mvave = movingmean(real(I_R),1001);

    # I_SE_mvave = movingmean(real(I_SE),5001);
    # I_L_mvave = movingmean(real(I_L),5001);
    # I_R_mvave = movingmean(real(I_R),5001);

    ref_some = [0.1, 0.3, 1.0, 3.0, 6.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    num0 = length(ref_some)
    time_some = zeros(Float64,num0)
    I_SE_some = zeros(Float64,num0)
    I_B_some = zeros(Float64,num0)
    I_L_some = zeros(Float64,num0)
    I_R_some = zeros(Float64,num0)
    Drelnuk_some = zeros(Float64,num0)
    I_SE_mvave_some = zeros(Float64,num0)
    I_L_mvave_some = zeros(Float64,num0)
    I_R_mvave_some = zeros(Float64,num0)
    sigma_c_some = zeros(Float64,num0)
    Drelpinuk_some = zeros(Float64,num0)
    for jj = 1:num0
        ind_some = argmin(abs.(time*GammaLR .- ref_some[jj]))
        time_some[jj] = time[ind_some]
        I_SE_some[jj] = real(I_SE[ind_some])
        I_B_some[jj] = real(I_B[ind_some])
        I_L_some[jj] = real(I_L[ind_some])
        I_R_some[jj] = real(I_R[ind_some])
        Drelnuk_some[jj] = real(Drelnuk[ind_some])
        I_SE_mvave_some[jj] = real(I_SE_mvave[ind_some])
        I_L_mvave_some[jj] = real(I_L_mvave[ind_some])
        I_R_mvave_some[jj] = real(I_R_mvave[ind_some])
        sigma_c_some[jj] = real(sigma_c[ind_some])
        Drelpinuk_some[jj] = real(Drelpinuk[ind_some])
    end

    p1 = plot(log10.(time[2:end]*GammaLR),log10.(real(sigma[2:end])),color=:black,lw=5,label=L"\sigma")
    plot!(log10.(time*GammaLR),log10.(real(I_SE)),color=:red,lw=3,label=L"I_{SB}")
    plot!(log10.(time*GammaLR),log10.(real(I_B)),color=:blue,lw=3,label=L"I_{B}")
    plot!(log10.(time*GammaLR),log10.(real(I_L)),color=:green,lw=3,label=L"I_{L}")
    plot!(log10.(time*GammaLR),log10.(real(I_R)),color=:orange,lw=3,label=L"I_{R}")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(Drelnuk[2:end])),color=:purple,lw=3,label=L"D_{env}")

    plot!(log10.(time_some*GammaLR),log10.(real(I_SE_some)),color=:red,lw=0,markershape=:circle,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(I_B_some)),color=:blue,lw=0,markershape=:rect,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(I_L_some)),color=:green,lw=0,markershape=:utriangle,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(I_R_some)),color=:orange,lw=0,markershape=:dtriangle,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(Drelnuk_some)),color=:purple,lw=0,markershape=:pentagon,ms=8)

    xlims!((-1.1,0.7))
    ylims!((-3,1))
    # ylims!((-6,1))
    plot!(legend=:none)

    p2 = plot(log10.(time[2:end]*GammaLR),log10.(real(sigma[2:end])),color=:black,lw=5,label=L"\sigma")
    plot!(log10.(time*GammaLR),log10.(real(I_SE_mvave)),color=:red,lw=3,label=L"I_{SB}")
    plot!(log10.(time*GammaLR),log10.(real(I_B)),color=:blue,lw=3,label=L"I_{B}")
    plot!(log10.(time*GammaLR),log10.(real(I_L_mvave)),color=:green,lw=3,label=L"I_{L}")
    plot!(log10.(time*GammaLR),log10.(real(I_R_mvave)),color=:orange,lw=3,label=L"I_{R}")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(Drelnuk[2:end])),color=:purple,lw=3,label=L"D_{env}")

    plot!(log10.(time_some*GammaLR),log10.(real(I_SE_mvave_some)),color=:red,lw=0,markershape=:star5,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(I_B_some)),color=:blue,lw=0,markershape=:rect,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(I_L_mvave_some)),color=:green,lw=0,markershape=:+,ms=12)
    plot!(log10.(time_some*GammaLR),log10.(real(I_R_mvave_some)),color=:orange,lw=0,markershape=:x,ms=12)
    plot!(log10.(time_some*GammaLR),log10.(real(Drelnuk_some)),color=:purple,lw=0,markershape=:pentagon,ms=8)

    xlims!((0.5,4.0))
    ylims!((-1,3))
    # ylims!((-1,3))
    plot!(legend=:none)

    p3 = plot(log10.(time[2:end]*GammaLR),log10.(real(sigma[2:end])),color=:black,lw=5,label=L"\sigma")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(sigma_c[2:end])),color=:grey,lw=3,label=L"\tilde{\sigma}")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(Drelpinuk[2:end])),color=:cyan,lw=3,label=L"\tilde{D}_{env}")

    plot!(log10.(time_some*GammaLR),log10.(real(sigma_c_some)),color=:grey,lw=0,markershape=:diamond,ms=8)
    plot!(log10.(time_some*GammaLR),log10.(real(Drelpinuk_some)),color=:cyan,lw=0,markershape=:hexagon,ms=8)

    ylims!((-4,3))
    plot!(legend=:none)

    plot(p1,p2,p3,layout=(1,3),size=(1100,400),dpi=600)
    # plot!(yticks=[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])

end

function averagecorrelationsregimeIII(K::Int64,betaL::Float64,betaR::Float64,muL::Float64,muR::Float64)

    # array_Gamma = [10.0^(-2), 10.0^(-1.5), 10.0^(-1), 10.0^(-0.5), 1.0, 10.0^(0.5), 10.0, 10.0^(1.5), 10.0^2]
    array_Gamma = [10.0^(-1), 10.0^(-0.5), 10.0^(0), 10.0^(0), 10.0^(0.5), 10.0]
    # array_Gamma = [10.0^(-1), 10.0^(-0.5), 10.0^(0)]
    array_W = [4, 4, 4, 20, 20, 20]
    # array_W = [4, 4, 4]
    tt_ref0 = 10.0^4
    tt_ref1 = 10.0^6 #10.0^(4.5)
    array_tt = 5*tt_ref1./array_Gamma

    array_I_SE = zeros(Float64,length(array_Gamma))
    array_I_B = zeros(Float64,length(array_Gamma))
    array_I_L = zeros(Float64,length(array_Gamma))
    array_I_R = zeros(Float64,length(array_Gamma))
    array_Drelnuk = zeros(Float64,length(array_Gamma))
    array_Drelpinuk = zeros(Float64,length(array_Gamma))

    array_sigma_d = zeros(Float64,length(array_Gamma))
    array_sigma_c = zeros(Float64,length(array_Gamma))

    for jj = 1:length(array_Gamma)

        Gamma = array_Gamma[jj]
        W = array_W[jj]
        tt = array_tt[jj]

        # time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR = calculatequantities2(K,W,0.0,betaL,betaR,Gamma,Gamma,muL,muR,tt,11) #501
        time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, Drelnuk, betaQL, betaQR, matCL, matCR, sigma_c2, Drelpinuk2, E_L, E_R, E_tot, N_L, N_R, boundL, boundR, Evariance_L, Evariance_R, EvarianceGibbs_L, EvarianceGibbs_R, Nvariance_L, Nvariance_R, NvarianceGibbs_L, NvarianceGibbs_R, Drel_rhoL_piL, Drel_rhoR_piR, Drel_rhoL_piL_ratio, Drel_rhoR_piR_ratio = calculatequantities4(K,K,W,betaL,betaR,Gamma,Gamma,muL,muR,tt,201) #11

        tt0 = argmin(abs.(time*Gamma.-tt_ref0))
        if time[tt0] < tt_ref0
           tt0 = tt0 + 1
        end

        tt1 = argmin(abs.(time*Gamma.-tt_ref1))
        # if time[tt1] < tt_ref1
           # tt1 = tt1 + 1
        # end

        array_I_SE[jj] = mean(real(I_SE[tt0:tt1]))
        array_I_B[jj] = mean(real(I_B[tt0:tt1]))
        array_I_L[jj] = mean(real(I_L[tt0:tt1]))
        array_I_R[jj] = mean(real(I_R[tt0:tt1]))
        array_Drelnuk[jj] = mean(real(Drelnuk[tt0:tt1]))
        array_Drelpinuk[jj] = mean(Drelpinuk2[tt0:tt1])

        # sigma_d = real(I_SE + I_B + I_L + I_R + Drelnuk)
        # sigma_c = real(I_SE + I_B + I_L + I_R + Drelpinuk2)

        # array_sigma_d[jj] = mean(sigma_d[tt0:tt1]/(2*log(2)+2*K*log(2)+2*K*log(2)+2*K*log(2)+boundL[1]+boundR[1]))
        # array_sigma_c[jj] = mean(sigma_c[tt0:tt1] ./ (2*log(2) .+ 2*K*log(2) .+ 2*K*log(2) .+ 2*K*log(2) .+ boundL[tt0:tt1] .+ boundR[tt0:tt1]))

    end

    return array_Gamma, array_W, array_I_SE, array_I_B, array_I_L, array_I_R, array_Drelnuk, array_Drelpinuk

end

function plot_averagecorrelationsregimeIII_each(array_Gamma, array_I_SE, array_I_B, array_I_L, array_I_R, array_Drelnuk, array_Drelpinuk)

    # plot!(log10.(array_Gamma[1:3]),array_I_SE[1:3]*2,color=:red,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle")
    # plot!(log10.(array_Gamma[4:6]),array_I_SE[4:6]*2,color=:blue,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle")

    # plot!(log10.(array_Gamma[1:3]),array_I_B[1:3]*2,color=:red,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")
    # plot!(log10.(array_Gamma[4:6]),array_I_B[4:6]*2,color=:blue,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")

    # plot!(log10.(array_Gamma[1:3]),array_I_L[1:3]*2,color=:red,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    # plot!(log10.(array_Gamma[4:6]),array_I_L[4:6]*2,color=:blue,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    
    # plot!(log10.(array_Gamma[1:3]),array_I_R[1:3]*2,color=:red,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")
    # plot!(log10.(array_Gamma[4:6]),array_I_R[4:6]*2,color=:blue,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")

    plot(log10.(array_Gamma[1:3]),array_Drelnuk[1:3],color=:red,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")
    plot!(log10.(array_Gamma[4:6]),array_Drelnuk[4:6],color=:blue,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_Drelpinuk[1:3]),color=:red,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_Drelpinuk[4:6]),color=:blue,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    
    xlims!((-1.1,1.1))
    # ylims!((-0.05,1.3))
    # plot!(aspect_ratio=0.5)
    plot!(legend=:none)

    # plot(p1,p2,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

    # plot(p4,p5,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

end

function plot_averagecorrelationsregimeIII(array_Gamma, array_I_SE, array_I_B, array_I_L, array_I_R, array_Drelnuk, array_Drelpinuk)

    value_x = [-2, 2]
    value_y = [1, 1]
    plot(value_x, value_y, lw=2, ls=:dash, color=:black, framestyle = :box)

    plot!(log10.(array_Gamma[1:3]),array_I_SE[1:3]*2,color=:red,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle")
    plot!(log10.(array_Gamma[4:6]),array_I_SE[4:6]*2,color=:blue,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle")

    plot!(log10.(array_Gamma[1:3]),array_I_B[1:3]*2,color=:red,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")
    plot!(log10.(array_Gamma[4:6]),array_I_B[4:6]*2,color=:blue,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")

    plot!(log10.(array_Gamma[1:3]),array_I_L[1:3]*2,color=:red,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    plot!(log10.(array_Gamma[4:6]),array_I_L[4:6]*2,color=:blue,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    
    plot!(log10.(array_Gamma[1:3]),array_I_R[1:3]*2,color=:red,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")
    plot!(log10.(array_Gamma[4:6]),array_I_R[4:6]*2,color=:blue,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_Drelnuk[1:3],color=:red,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_Drelnuk[4:6],color=:blue,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_Drelpinuk[1:3]),color=:red,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_Drelpinuk[4:6]),color=:blue,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    
    xlims!((-1.1,1.1))
    ylims!((-0.05,1.3))
    # plot!(aspect_ratio=0.5)
    plot!(legend=:none)

    # plot(p1,p2,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

    # plot(p4,p5,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

end

function plot_averagecorrelationsregimeIII_log(array_Gamma, array_I_SE, array_I_B, array_I_L, array_I_R, array_Drelnuk, array_Drelpinuk)

    # plot(log10.(array_Gamma[1:3]),log10.(array_sigma_c[1:3]),color=:red,marker=(:circle,8),lw=3,label=L"\langle \sigma_c \rangle",framestyle = :box)
    # plot!(log10.(array_Gamma[4:6]),log10.(array_sigma_c[4:6]),color=:blue,marker=(:circle,8),lw=3,label=L"\langle \sigma_c \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_sigma_d[1:3]),color=:red,marker=(:star4,8),lw=3,label=L"\langle \sigma_d \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_sigma_d[4:6]),color=:blue,marker=(:star4,8),lw=3,label=L"\langle \sigma_d \rangle")

    plot(log10.(array_Gamma[1:3]),log10.(array_I_SE[1:3]*2),color=:red,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle", framestyle = :box)
    plot!(log10.(array_Gamma[4:6]),log10.(array_I_SE[4:6]*2),color=:blue,marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle")

    plot!(log10.(array_Gamma[1:3]),log10.(array_I_B[1:3]*2),color=:red,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")
    plot!(log10.(array_Gamma[4:6]),log10.(array_I_B[4:6]*2),color=:blue,marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")

    plot!(log10.(array_Gamma[1:3]),log10.(array_I_L[1:3]*2),color=:red,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    plot!(log10.(array_Gamma[4:6]),log10.(array_I_L[4:6]*2),color=:blue,marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    
    plot!(log10.(array_Gamma[1:3]),log10.(array_I_R[1:3]*2),color=:red,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")
    plot!(log10.(array_Gamma[4:6]),log10.(array_I_R[4:6]*2),color=:blue,marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_Drelnuk[1:3]),color=:red,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_Drelnuk[4:6]),color=:blue,marker=(:square,8),lw=3,label=L"\langle D_{env} \rangle")

    # plot!(log10.(array_Gamma[1:3]),log10.(array_Drelpinuk[1:3]),color=:red,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    # plot!(log10.(array_Gamma[4:6]),log10.(array_Drelpinuk[4:6]),color=:blue,marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")
    
    xlims!((-1.1,1.1))
    ylims!((-4,0))
    plot!(aspect_ratio=0.5)
    plot!(legend=:none)

    # plot(p1,p2,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

    # plot(p4,p5,p3,layout=(1,3),size=(800,300),dpi=600)
    # plot!(legend=:none)

end

function test_averageDrelpinukregimeIII()

    array_Gamma = [10.0^(-2), 10.0^(-1.5), 10.0^(-1), 10.0^(-0.5), 1.0, 10.0^(0.5), 10.0, 10.0^(1.5), 10.0^2]

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL32R32muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL32R32muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[8]

    plot(log10.(time*Gamma),real(Drelpinuk))

end

function averageDrelpinukregimeIII()

    array_Gamma = [10.0^(-2), 10.0^(-1.5), 10.0^(-1), 10.0^(-0.5), 1.0, 10.0^(0.5), 10.0, 10.0^(1.5), 10.0^2]
    tt_ref0 = 10.0^3
    tt_ref1 = 10.0^(3.5)
    array_Drelpinuk = zeros(Float64,length(array_Gamma))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL001R001muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL001R001muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[1]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[1] = mean(real(Drelpinuk[tt0:tt1]))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaLm32Rm32muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaLm32Rm32muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[2]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[2] = mean(real(Drelpinuk[tt0:tt1]))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL01R01muL1R1tf50000Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL01R01muL1R1tf50000Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[3]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[3] = mean(real(Drelpinuk[tt0:tt1]))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaLm05Rm05muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaLm05Rm05muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[4]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[4] = mean(real(Drelpinuk[tt0:tt1]))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL1R1muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL1R1muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[5]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[5] = mean(real(Drelpinuk[tt0:tt1]))

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL05R05muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL05R05muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[6]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[6] = mean(real(Drelpinuk[tt0:tt1]))

    tt_ref0 = 10.0^(3.5)
    tt_ref1 = 10.0^(4)

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL10R10muL1R1tf100000Nt100001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL10R10muL1R1tf100000Nt100001.jld")["Drelpinuk"];
    Gamma = array_Gamma[7]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[7] = mean(real(Drelpinuk[tt0:tt1]))

    tt_ref0 = 10.0^(3.5)
    tt_ref1 = 10.0^(4)

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL32R32muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL32R32muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[8]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[8] = mean(real(Drelpinuk[tt0:tt1]))

    tt_ref0 = 10.0^(4)
    tt_ref1 = 10.0^(4.5)

    time = load("data_calculatequantities2_K128W20betaL1R01GammaL100R100muL1R1Nt50001.jld")["time"];
    Drelpinuk = load("data_calculatequantities2_K128W20betaL1R01GammaL100R100muL1R1Nt50001.jld")["Drelpinuk"];
    Gamma = array_Gamma[9]
    tt0 = argmin(abs.(time*Gamma.-tt_ref0))
    if time[tt0] < tt_ref0
       tt0 = tt0 + 1
    end
    tt1 = argmin(abs.(time*Gamma.-tt_ref1))
    array_Drelpinuk[9] = mean(real(Drelpinuk[tt0:tt1]))

    return array_Gamma, array_Drelpinuk

end

################ basis functions for obs

function swap!(aa::Int64,bb::Int64,vec::Vector{Int64})
    spot = vec[aa]
    vec[aa] = vec[bb]
    vec[bb] = spot
end

function number_sumoffactorial(size::Int64)

    length0 = zeros(Int64,size)
    length0[1] = 1
    for jj = 2:size
        # length0[jj] = Int64(3/2*factorial(jj))
        length0[jj] = prod(3:2:2*jj-1)
    end

    return sum(length0)

end

function populationinbath(size::Int64)

    # look at Eq. (30) in arXiv:1710.09248

    # number of two point correlation terms for system size = 2,...,size
    length0 = zeros(Int64,size)
    length0[1] = 1
    for jj = 2:size
        # length0[jj] = Int64(3/2*factorial(jj))
        length0[jj] = prod(3:2:2*jj-1)
    end
    mat0 = zeros(Int64,sum(length0),size*2)

    # system size =1
    mat0[1,1:2] = [1 2]

    # system size =2
    mat0[length0[1]+1:sum(length0[1:2]),1:2*2] = [1 2 3 4; 1 3 2 4; 1 4 2 3]

    # system size >2
    for jj = 3:size

        mat1 = mat0[sum(length0[1:jj-2])+1:sum(length0[1:jj-1]),1:2*(jj-1)]
        mat0[sum(length0[1:jj-1])+1:sum(length0[1:jj-1])+length0[jj-1],1:2*(jj-1)] .= mat1
        mat0[sum(length0[1:jj-1])+1:sum(length0[1:jj-1])+length0[jj-1],2*jj-1] .= 2*jj-1
        mat0[sum(length0[1:jj-1])+1:sum(length0[1:jj-1])+length0[jj-1],2*jj] .= 2*jj

        for ii = 1:2*jj-2 #1:jj-1
            # test0 = collect(1:2*(jj-1))
            # deleteat!(test0,2*jj-1-2*ii)
            # push!(test0,2*jj-1)
            #
            test0 = collect(1:2*(jj-1))
            deleteat!(test0,2*jj-1-ii)
            push!(test0,2*jj-1)

            for kk = 1:length0[jj-1]
                mat0[sum(length0[1:jj-1])+length0[jj-1]*ii+kk,1:2*(jj-1)] = test0[mat1[kk,:]]
                mat0[sum(length0[1:jj-1])+length0[jj-1]*ii+kk,2*jj-1:2*jj] = [2*jj-1-ii 2*jj] #[2*jj-1-2*ii 2*jj]
            end
        end
    end

    # return mat0

    # remove <c_i c_j>=<c_i^+ c_j^+>=0
    mat2 = zeros(Int64,sum(length0),size*2)
    vec2 = zeros(Int64,sum(length0))
    check = 0
    ind = 0
    size_ind = 0
    for jj = 1:sum(length0)
        check = 0
        size_ind = size
        for ii = 1:size
            if isodd(mat0[jj,2*ii-1]*mat0[jj,2*ii])
               break
            elseif mat0[jj,2*ii-1]*mat0[jj,2*ii] == 0
               check = size
               size_ind = ii-1
               break
            end
            check += 1
        end
        if check == size
           ind += 1
           mat2[ind,:] = mat0[jj,:]
           vec2[ind] = size_ind
        end
    end
    mat2 = mat2[1:ind,:]
    vec2 = vec2[1:ind]

    # coefficient (-1)^P
    vec0 = zeros(Int64,size*2)
    vec1 = zeros(Int64,ind)
    for jj = 1:ind
        check = 0
        vec0 .= mat2[jj,:]
        for ii = 1:size*2
            if vec0[ii] == ii
               continue
            elseif vec0[ii] == 0
               break
            end
            switch = findfirst(x->x==ii,vec0)
            swap!(ii,switch,vec0)
            check += 1
        end
        vec1[jj] = (-1)^(check)
    end

    # return mat2

    # [even,odd] -> [odd,even]*(-1)
    vec3 = ones(Int64,ind)
    for jj = 1:ind
        for ii = 1:size
            if iseven(mat2[jj,2*ii-1]) && isodd(mat2[jj,2*ii])
               spot = mat2[jj,2*ii-1]
               mat2[jj,2*ii-1] = mat2[jj,2*ii]
               mat2[jj,2*ii] = spot
               vec3[jj] *= -1
            end
        end
    end

    # a_1 a_2 -> c_1^+ c_1
    mat3 = copy(mat2)
    for jj = 1:ind
        for ii = 1:size*2
            mat3[jj,ii] = ceil(Int64,mat3[jj,ii]/2)
        end
    end

    # give the set of ...
    # indices for the probability
    # the coefficient (-1)^P
    # the system size of the index set
    # (-1) for replacing c c^+ to c^+ c
    return mat3, vec1, vec2, vec3

    # return mat0, mat2, vec1

    # mat1 = [1 2 3 4; 1 3 2 4; 1 4 2 3]
    # mat0[1:length0[2],1:2*2] = mat1
    #
    # mat0[sum(length0[1:2])+1:sum(length0[1:2])+length0[2],1:2*2] = mat1
    # mat0[sum(length0[1:2])+1,2*2+1:2*2+2] = [5 6]
    # mat0[sum(length0[1:2])+2,2*2+1:2*2+2] = [5 6]
    # mat0[sum(length0[1:2])+3,2*2+1:2*2+2] = [5 6]
    #
    # test0 = collect(1:2*2)
    # deleteat!(test0,3)
    # push!(test0,5)
    # mat0[sum(length0[1:2])+length0[2]+1,1:2*2] = test0[mat1[1,:]]
    # mat0[sum(length0[1:2])+length0[2]+1,2*2+1:2*2+2] = [3 6]
    # mat0[sum(length0[1:2])+length0[2]+2,1:2*2] = test0[mat1[2,:]]
    # mat0[sum(length0[1:2])+length0[2]+2,2*2+1:2*2+2] = [3 6]
    # mat0[sum(length0[1:2])+length0[2]+3,1:2*2] = test0[mat1[3,:]]
    # mat0[sum(length0[1:2])+length0[2]+3,2*2+1:2*2+2] = [3 6]
    #
    # test0 = collect(1:2*2)
    # deleteat!(test0,1)
    # push!(test0,5)
    # mat0[sum(length0[1:2])+length0[2]+4,1:2*2] = test0[mat1[1,:]]
    # mat0[sum(length0[1:2])+length0[2]+4,2*2+1:2*2+2] = [1 6]
    # mat0[sum(length0[1:2])+length0[2]+5,1:2*2] = test0[mat1[2,:]]
    # mat0[sum(length0[1:2])+length0[2]+5,2*2+1:2*2+2] = [1 6]
    # mat0[sum(length0[1:2])+length0[2]+6,1:2*2] = test0[mat1[3,:]]
    # mat0[sum(length0[1:2])+length0[2]+6,2*2+1:2*2+2] = [1 6]
    #
    # return mat0

end

function calculate_Sobs_test3(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2 or small
    # K = 6

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_fluctuatedt!(K,W,t_flu,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)
    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)
    eigval_Ct_E = zeros(Float64,2*K)

    pLR_NE_E = ones(Int64,K+1,binomial(K,floor(Int64,K/2)))*(-1)
    pLR_NE_V = zeros(Int64,K+1,binomial(K,floor(Int64,K/2)))
    pLR_NE_deg = zeros(Int64,2^K,3)

    p_each = zeros(Float64,2*K+1,binomial(2*K+1,K))
    ptotal_sys_NLEL_NRER = zeros(Float64,2,K+1,binomial(K,floor(Int64,K/2)),K+1,binomial(K,floor(Int64,K/2)))
    # ptotal_sys_NLEL_NRER_t0 = zeros(Float64,2,K+1,binomial(K,floor(Int64,K/2)),K+1,binomial(K,floor(Int64,K/2)))

    pL_each = zeros(Float64,K,binomial(K,floor(Int64,K/2)))
    pL_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pL_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    pR_each = zeros(Float64,K,binomial(K,floor(Int64,K/2)))
    pR_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pR_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    pLR_each = zeros(Float64,2*K+1,binomial(2*K+1,K))
    pLR_NLEL_NRER = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)),K+1,binomial(K,floor(Int64,K/2)))

    SobsL = zeros(Float64,Nt)
    SobsR = zeros(Float64,Nt)
    SobsE = zeros(Float64,Nt)
    Sobs = zeros(Float64,Nt)
    DrelL_obs = zeros(Float64,Nt)
    DrelR_obs = zeros(Float64,Nt)
    Ssys = zeros(Float64,Nt)
    sigmaobs = zeros(Float64,Nt)
    Iobs_SE = zeros(Float64,Nt)
    Iobs_B = zeros(Float64,Nt)

    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    vNE_E = zeros(Float64,Nt)
    vNE = zeros(Float64,Nt)
    dCt = zeros(ComplexF64,2*K+1,2*K+1)
    QL = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)

    # pL_EN_E,V for N=0 and so E=0
    pLR_NE_E[1,1] = 0
    pLR_NE_V[1,1] = 1
    indLR = 0

    # pLR_NE_E,V for N>=1
    for jjN = 1:K
        label = collect(combinations(1:K,jjN))
        for jjNlabel = 1:length(label)

            pLR_NE_E[1+jjN,jjNlabel] = sum(label[jjNlabel])
            pLR_NE_V[1+jjN,jjNlabel] = 1

            # check degenracy
            for jjcheck = 1:jjNlabel-1
                if pLR_NE_E[1+jjN,jjcheck] == pLR_NE_E[1+jjN,jjNlabel] #&& pL_NE_V[1+jjN,jjcheck] > 0
                   indLR += 1
                   pLR_NE_deg[indLR,1] = jjN
                   pLR_NE_deg[indLR,2] = jjNlabel
                   pLR_NE_deg[indLR,3] = jjcheck
                   pLR_NE_V[1+jjN,jjcheck] += 1
                   pLR_NE_V[1+jjN,jjNlabel] = -1
                   break
                end
            end

        end
    end
    pLR_NE_deg = pLR_NE_deg[1:indLR,:]

    #
    list_index, list_coeffP, list_size, list_coeffswap = populationinbath(2*K+1);

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # total
        lambda, eigvec_Ct = eigen(Ct)
        eigval_Ct .= real.(lambda)

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        # p_each
        p_each .= 0.0

        # N=2K+1
        ind0 = findfirst(x->x==2*K+1,list_size)
        ind1 = findlast(x->x==2*K+1,list_size)
        for jj = ind0:ind1
            prob = Ct[list_index[jj,1],list_index[jj,2]]
            for ii = 2:2*K+1
                prob *= Ct[list_index[jj,2*ii-1],list_index[jj,2*ii]]
            end
            p_each[2*K+1,1] += real(prob)*list_coeffP[jj]*list_coeffswap[jj]
        end

        # 1 <= N <= 2K
        for jjN = 2*K:-1:1
            label = collect(combinations(1:2*K+1,jjN))
            for jjNlabel = 1:length(label)

                ind0 = findfirst(x->x==jjN,list_size)
                ind1 = findlast(x->x==jjN,list_size)
                label0 = label[jjNlabel]

                for jj = ind0:ind1

                    list_index0 = list_index[jj,1:jjN*2]
                    list_index1 = copy(list_index0)
                    for ii = 1:jjN
                        list_index1[findfirst(x->x==ii,list_index0)] = label0[ii]
                        list_index1[findlast(x->x==ii,list_index0)] = label0[ii]
                    end

                    prob = Ct[list_index1[1],list_index1[2]]
                    for ii = 2:jjN
                        prob *= Ct[list_index1[2*ii-1],list_index1[2*ii]]
                    end
                    p_each[jjN,jjNlabel] += real(prob)*list_coeffP[jj]*list_coeffswap[jj]

                end

            end
        end

        # pL_each and pR_each
        pL_each .= 0.0
        pR_each .= 0.0
        for jjN = K:-1:1
            label = collect(combinations(1:2*K+1,jjN))
            label_L_part = collect(combinations(2:K+1,jjN))
            label_R_part = collect(combinations(K+2:2*K+1,jjN))
            for jj1 = 1:length(label_L_part) #length(label_R_part)

                indL = 0
                for jj2 = 1:length(label)
                    if label_L_part[jj1] == label[jj2]
                       indL = jj2
                       break
                    end
                end
                pL_each[jjN,jj1] = p_each[jjN,indL]

                indR = 0
                for jj2 = 1:length(label)
                    if label_R_part[jj1] == label[jj2]
                       indR = jj2
                       break
                    end
                end
                pR_each[jjN,jj1] = p_each[jjN,indR]

            end
        end

        # pLR_each
        pLR_each .= 0.0
        for jjN = 2*K:-1:1
            label = collect(combinations(1:2*K+1,jjN))
            label_LR_part = collect(combinations(2:2*K+1,jjN))
            for jj1 = 1:length(label_LR_part)
                ind = 0
                for jj2 = 1:length(label)
                    if label_LR_part[jj1] == label[jj2]
                       ind = jj2
                       break
                    end
                end
                pLR_each[jjN,jj1] = p_each[jjN,ind]
            end
        end

        # pL_NE and pR_NE
        pL_NE .= 0.0
        pR_NE .= 0.0

        # NL,NR=0
        pL_NE[1,1] = 1
        pR_NE[1,1] = 1
        for jj = 1:K
            pL_NE[1,1] = pL_NE[1,1] + sum(pL_each[jj,:])*(-1)^(jj)
            pR_NE[1,1] = pR_NE[1,1] + sum(pR_each[jj,:])*(-1)^(jj)
        end

        # NL,NR>=1
        for jjN = 1:K
            label = collect(combinations(1:K,jjN))
            for jj1 = 1:length(label)
                label1 = label[jj1]
                for jj2 = jjN:K
                    label2 = collect(combinations(1:K,jj2))
                    for jj3 = 1:length(label2)
                        if issubset(label1,label2[jj3])
                           pL_NE[jjN+1,jj1] = pL_NE[jjN+1,jj1] + pL_each[jj2,jj3]*(-1)^(jj2-jjN)
                           pR_NE[jjN+1,jj1] = pR_NE[jjN+1,jj1] + pR_each[jj2,jj3]*(-1)^(jj2-jjN)
                        end
                    end
                end

                for jjdeg = 1:indLR
                    if jjN == pLR_NE_deg[jjdeg,1] && jj1 == pLR_NE_deg[jjdeg,2]
                       pL_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pL_NE[1+jjN,jj1]
                       pL_NE[1+jjN,jj1] = 0.0
                       pR_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pR_NE[1+jjN,jj1]
                       pR_NE[1+jjN,jj1] = 0.0
                       break #
                    end
                end

            end
        end

        # pLR_NLEL_NRER
        pLR_NLEL_NRER .= 0.0

        # NL+NR=0
        pLR_NLEL_NRER[1,1,1,1] = 1
        for jj = 1:2*K
            pLR_NLEL_NRER[1,1,1,1] = pLR_NLEL_NRER[1,1,1,1] + sum(pLR_each[jj,:])*(-1)^(jj)
        end

        # NL+NR >= 1
        for jjN = 1:2*K
            label = collect(combinations(1:2*K,jjN))
            for jj1 = 1:length(label)
                label1 = label[jj1]

                # L
                indNL = 1
                indEL = 1
                if label1[1] <= K
                   labelL = label1[findfirst(x->x<=K,label1):findlast(x->x<=K,label1)]
                   NL = length(labelL)
                   labelL_all = collect(combinations(1:K,NL))
                   indNL = NL+1
                   for jjL = 1:length(labelL_all)
                       if labelL == labelL_all[jjL]
                          indEL = jjL
                          break
                       end
                   end
                end

                # R
                indNR = 1
                indER = 1
                if label1[end] >= K+1
                   labelR = label1[findfirst(x->x>=K+1,label1):findlast(x->x>=K+1,label1)]
                   labelR .= labelR .- K
                   NR = length(labelR)
                   labelR_all = collect(combinations(1:K,NR))
                   indNR = NR+1
                   for jjR = 1:length(labelR_all)
                       if labelR == labelR_all[jjR]
                          indER = jjR
                          break
                       end
                   end
                end

                for jj2 = jjN:2*K
                    label2 = collect(combinations(1:2*K,jj2))
                    for jj3 = 1:length(label2)
                        if issubset(label1,label2[jj3])
                           pLR_NLEL_NRER[indNL,indEL,indNR,indER] = pLR_NLEL_NRER[indNL,indEL,indNR,indER] + pLR_each[jj2,jj3]*(-1)^(jj2-jjN)
                        end
                    end
                end

                indEL1 = 0
                for jjdeg = 1:indLR
                    if indNL == pLR_NE_deg[jjdeg,1] && indEL == pLR_NE_deg[jjdeg,2]
                       indEL1 = pLR_NE_deg[jjdeg,3]
                       break
                    end
                end

                indER1 = 0
                for jjdeg = 1:indLR
                    if indNR == pLR_NE_deg[jjdeg,1] && indER == pLR_NE_deg[jjdeg,2]
                       indER1 = pLR_NE_deg[jjdeg,3]
                       break
                    end
                end

                if indEL1 != 0 && indER1 != 0
                   pLR_NLEL_NRER[indNL,indEL1,indNR,indER1] += pLR_NLEL_NRER[indNL,indEL,indNR,indER]
                   pLR_NLEL_NRER[indNL,indEL,indNR,indER] = 0.0
                elseif indEL1 != 0 && indER1 == 0
                   pLR_NLEL_NRER[indNL,indEL1,indNR,indER] += pLR_NLEL_NRER[indNL,indEL,indNR,indER]
                   pLR_NLEL_NRER[indNL,indEL,indNR,indER] = 0.0
                elseif indEL1 == 0 && indER1 != 0
                   pLR_NLEL_NRER[indNL,indEL,indNR,indER1] += pLR_NLEL_NRER[indNL,indEL,indNR,indER]
                   pLR_NLEL_NRER[indNL,indEL,indNR,indER] = 0.0
                # elseif indEL1 == 0 && indER1 == 0
                end

            end
        end

        # ptotal_sys_NLEL_NRER
        ptotal_sys_NLEL_NRER .= 0.0

        # Ntotal=0
        ptotal_sys_NLEL_NRER[1,1,1,1,1] = 1
        for jj = 1:2*K+1
            ptotal_sys_NLEL_NRER[1,1,1,1,1] = ptotal_sys_NLEL_NRER[1,1,1,1,1] + sum(p_each[jj,:])*(-1)^(jj)
        end

        # Ntotal >= 1
        for jjN = 1:2*K+1
            label = collect(combinations(1:2*K+1,jjN))
            for jj1 = 1:length(label)
                label1 = label[jj1]

                # system
                indsys = 1
                if label1[1] == 1
                   indsys = 2
                end

                # L
                indNL = 1
                indEL = 1
                if label1 != [1] && label1[indsys] >= 2 && label1[indsys] <= K+1
                   labelL = label1[findfirst(x->x>=2&&x<=K+1,label1):findlast(x->x>=2&&x<=K+1,label1)]
                   labelL .= labelL .- 1
                   NL = length(labelL)
                   labelL_all = collect(combinations(1:K,NL))
                   indNL = NL+1
                   for jjL = 1:length(labelL_all)
                       if labelL == labelL_all[jjL]
                          indEL = jjL
                          break
                       end
                   end
                end

                # R
                indNR = 1
                indER = 1
                if label1[end] >= K+2
                   labelR = label1[findfirst(x->x>=K+2,label1):findlast(x->x>=K+2,label1)]
                   labelR .= labelR .- (K+1)
                   NR = length(labelR)
                   labelR_all = collect(combinations(1:K,NR))
                   indNR = NR+1
                   for jjR = 1:length(labelR_all)
                       if labelR == labelR_all[jjR]
                          indER = jjR
                          break
                       end
                   end
                end

                for jj2 = jjN:2*K+1
                    label2 = collect(combinations(1:2*K+1,jj2))
                    for jj3 = 1:length(label2)
                        if issubset(label1,label2[jj3])
                           ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER] = ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER] + p_each[jj2,jj3]*(-1)^(jj2-jjN)
                        end
                    end
                end

                indEL1 = 0
                for jjdeg = 1:indLR
                    if indNL == pLR_NE_deg[jjdeg,1] && indEL == pLR_NE_deg[jjdeg,2]
                       indEL1 = pLR_NE_deg[jjdeg,3]
                       break
                    end
                end

                indER1 = 0
                for jjdeg = 1:indLR
                    if indNR == pLR_NE_deg[jjdeg,1] && indER == pLR_NE_deg[jjdeg,2]
                       indER1 = pLR_NE_deg[jjdeg,3]
                       break
                    end
                end

                if indEL1 != 0 && indER1 != 0
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL1,indNR,indER1] += ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER]
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER] = 0.0
                elseif indEL1 != 0 && indER1 == 0
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL1,indNR,indER] += ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER]
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER] = 0.0
                elseif indEL1 == 0 && indER1 != 0
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER1] += ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER]
                   ptotal_sys_NLEL_NRER[indsys,indNL,indEL,indNR,indER] = 0.0
                # elseif indEL1 == 0 && indER1 == 0
                end

            end
        end

        if tt == 1
           pL_NE_t0 .= pL_NE
           pR_NE_t0 .= pR_NE
        end

        # observational entropy and relative entropy
        for jjN = 1:K+1
            for jjNlabel = 1:binomial(K,jjN-1)

                if pLR_NE_V[jjN,jjNlabel] > 0
                   SobsL[tt] += pL_NE[jjN,jjNlabel]*(-log(pL_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   SobsR[tt] += pR_NE[jjN,jjNlabel]*(-log(pR_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   DrelL_obs[tt] += pL_NE[jjN,jjNlabel]*(log(pL_NE[jjN,jjNlabel])-log(pL_NE_t0[jjN,jjNlabel]))
                   DrelR_obs[tt] += pR_NE[jjN,jjNlabel]*(log(pR_NE[jjN,jjNlabel])-log(pR_NE_t0[jjN,jjNlabel]))
                end

            end
        end

        for jjNL = 1:K+1
            for jjNlabelL = 1:binomial(K,jjNL-1)
                if pLR_NE_V[jjNL,jjNlabelL] <= 0
                   continue
                end
                for jjNR = 1:K+1
                    for jjNlabelR = 1:binomial(K,jjNR-1)
                        if pLR_NE_V[jjNR,jjNlabelR] <= 0
                           continue
                        end

                        if abs(pLR_NLEL_NRER[jjNL,jjNlabelL,jjNR,jjNlabelR]) < 1e-15
                           continue
                        end
                        SobsE[tt] += pLR_NLEL_NRER[jjNL,jjNlabelL,jjNR,jjNlabelR]*(-log(pLR_NLEL_NRER[jjNL,jjNlabelL,jjNR,jjNlabelR])+log(pLR_NE_V[jjNL,jjNlabelL]*pLR_NE_V[jjNR,jjNlabelR]))

                        for jjNsys = 1:2

                            # if ptotal_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR] == 0.0
                            if abs(ptotal_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR]) < 1e-15
                               continue
                            end
                            Sobs[tt] += ptotal_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR]*(-log(ptotal_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR])+log(pLR_NE_V[jjNL,jjNlabelL]*pLR_NE_V[jjNR,jjNlabelR]))

                        end
                    end
                end
            end
        end

        # system
        Ssys[tt] = real(-Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1]))

        # entropy production of obs
        sigmaobs[tt] = Ssys[tt] - Ssys[1] + SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] + DrelL_obs[tt] + DrelR_obs[tt]

        # correlation
        Iobs_SE[tt] = Ssys[tt] - Ssys[1] + SobsE[tt] - SobsE[1] - (Sobs[tt] - Sobs[1])
        Iobs_B[tt] = SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] - (SobsE[tt] - SobsE[1])

        # vNE
        vNE_L[tt] = real(- sum(eigval_Ct_L.*log.(eigval_Ct_L)) - sum((1.0 .- eigval_Ct_L).*log.(1.0 .- eigval_Ct_L)))
        vNE_R[tt] = real(- sum(eigval_Ct_R.*log.(eigval_Ct_R)) - sum((1.0 .- eigval_Ct_R).*log.(1.0 .- eigval_Ct_R)))
        vNE[tt] = real(- sum(eigval_Ct.*log.(eigval_Ct)) - sum((1.0 .- eigval_Ct).*log.(1.0 .- eigval_Ct)))
        eigval_Ct_E .= real(eigvals(Ct[2:end,2:end]))
        vNE_E[tt] = real(- sum(eigval_Ct_E.*log.(eigval_Ct_E)) - sum((1.0 .- eigval_Ct_E).*log.(1.0 .- eigval_Ct_E)))

        #
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        betaQR[tt] = QR[tt]*betaR
        sigma[tt] = Ssys[tt] - Ssys[1] - betaQL[tt] - betaQR[tt]

    end

    # return ptotal_sys_NLEL_NRER, Sobs, vNE, Ct, pLR_NE_V
    return time, sigmaobs, Ssys, Sobs, SobsE, SobsL, SobsR, DrelL_obs, DrelR_obs, Iobs_SE, Iobs_B, sigma, vNE, vNE_E, vNE_L, vNE_R

end

function calculate_Sobs_test2(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2 or small
    # K = 6

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_fluctuatedt!(K,W,t_flu,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)
    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    pL_n = zeros(Float64,K)
    pL_n0 = zeros(Float64,K)
    pR_n = zeros(Float64,K)
    pR_n0 = zeros(Float64,K)
    p_n = zeros(Float64,2*K+1)
    p_n0 = zeros(Float64,2*K+1)

    pL_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pL_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    pR_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pR_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    p_sys_NLEL_NRER = zeros(Float64,2,K+1,binomial(K,floor(Int64,K/2)),K+1,binomial(K,floor(Int64,K/2)))
    p_NLEL_NRER = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)),K+1,binomial(K,floor(Int64,K/2)))

    pLR_NE_E = ones(Int64,K+1,binomial(K,floor(Int64,K/2)))*(-1)
    pLR_NE_V = zeros(Int64,K+1,binomial(K,floor(Int64,K/2)))
    pLR_NE_deg = zeros(Int64,2^K,3)

    SobsL = zeros(Float64,Nt)
    SobsR = zeros(Float64,Nt)
    DrelL_obs = zeros(Float64,Nt)
    DrelR_obs = zeros(Float64,Nt)
    Sobssys = zeros(Float64,Nt)
    sigmaobs = zeros(Float64,Nt)
    Sobs = zeros(Float64,Nt)
    SobsE = zeros(Float64,Nt)
    Iobs_SE = zeros(Float64,Nt)
    Iobs_B = zeros(Float64,Nt)

    dCt = zeros(ComplexF64,K*2+1)
    QL = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    vNE = zeros(Float64,Nt)
    vNE_E = zeros(Float64,Nt)
    Drel_vNE = zeros(ComplexF64,Nt)

    # pL_EN_E,V for N=0 and so E=0
    pLR_NE_E[1,1] = 0
    pLR_NE_V[1,1] = 1
    ind = 0

    # pLR_NE_E,V for N>=1
    for jjN = 1:K
        label = collect(combinations(1:K,jjN))
        for jjNlabel = 1:length(label)

            pLR_NE_E[1+jjN,jjNlabel] = sum(label[jjNlabel])
            pLR_NE_V[1+jjN,jjNlabel] = 1

            # check degenracy
            for jjcheck = 1:jjNlabel-1
                if pLR_NE_E[1+jjN,jjcheck] == pLR_NE_E[1+jjN,jjNlabel] #&& pL_NE_V[1+jjN,jjcheck] > 0
                   ind += 1
                   pLR_NE_deg[ind,1] = jjN
                   pLR_NE_deg[ind,2] = jjNlabel
                   pLR_NE_deg[ind,3] = jjcheck
                   pLR_NE_V[1+jjN,jjcheck] += 1
                   pLR_NE_V[1+jjN,jjNlabel] = -1
                   break
                end
            end

        end
    end
    pLR_NE_deg = pLR_NE_deg[1:ind,:]

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # total
        lambda, eigvec_Ct = eigen(Ct)
        eigval_Ct .= real.(lambda)

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        p_n .= real(diag(Ct))
        psys_n = p_n[1]
        pL_n .= p_n[2:K+1] #(abs.(eigvec_Ct_L).^2)*eigval_Ct_L
        pR_n .= p_n[K+2:2*K+1] #(abs.(eigvec_Ct_R).^2)*eigval_Ct_R

        # pL_EN and pR_EN
        pL_NE .= 0.0
        pR_NE .= 0.0

        # for N=0 and so E=0
        pL_n0 .= 1.0 .- pL_n
        pL_NE[1,1] = prod(pL_n0)
        pR_n0 .= 1.0 .- pR_n
        pR_NE[1,1] = prod(pR_n0)

        # pL_EN and pR_EN for N>=1
        for jjN = 1:K
            label = collect(combinations(1:K,jjN))
            for jjNlabel = 1:length(label)

                # L
                pL_n0 .= 1.0 .- pL_n
                pL_n0[label[jjNlabel]] = pL_n[label[jjNlabel]]
                pL_NE[1+jjN,jjNlabel] = prod(pL_n0)

                # R
                pR_n0 .= 1.0 .- pR_n
                pR_n0[label[jjNlabel]] = pR_n[label[jjNlabel]]
                pR_NE[1+jjN,jjNlabel] = prod(pR_n0)

                for jjdeg = 1:ind
                    if jjN == pLR_NE_deg[jjdeg,1] && jjNlabel == pLR_NE_deg[jjdeg,2]
                       pL_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pL_NE[1+jjN,jjNlabel]
                       pL_NE[1+jjN,jjNlabel] = 0.0
                       pR_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pR_NE[1+jjN,jjNlabel]
                       pR_NE[1+jjN,jjNlabel] = 0.0
                       break #
                    end
                end

            end
        end
        if tt == 1
           pL_NE_t0 .= pL_NE
           pR_NE_t0 .= pR_NE
        end

        # p_sys_NLEL_NRER
        p_sys_NLEL_NRER .= 0.0

        # Nsys=0,1
        p_sys_NLEL_NRER[1,:,:,:,:] .= 1-psys_n
        p_sys_NLEL_NRER[2,:,:,:,:] .= psys_n

        # NL=0
        pL_n0 .= 1.0 .- pL_n
        p_sys_NLEL_NRER[:,1,1,:,:] *= prod(pL_n0)

        # NL>=1
        for jjNL = 1:K
            labelL = collect(combinations(1:K,jjNL))
            for jjNlabelL = 1:length(labelL)

                pL_n0 .= 1.0 .- pL_n
                pL_n0[labelL[jjNlabelL]] = pL_n[labelL[jjNlabelL]]
                p_sys_NLEL_NRER[:,jjNL+1,jjNlabelL,:,:] *= prod(pL_n0)

                for jjdeg = 1:ind
                    if jjNL == pLR_NE_deg[jjdeg,1] && jjNlabelL == pLR_NE_deg[jjdeg,2]
                       p_sys_NLEL_NRER[:,jjNL+1,pLR_NE_deg[jjdeg,3],:,:] += p_sys_NLEL_NRER[:,jjNL+1,jjNlabelL,:,:]
                       p_sys_NLEL_NRER[:,jjNL+1,jjNlabelL,:,:] .= 0.0
                       break #
                    end
                end
            end
        end

        # NR=0
        pR_n0 .= 1.0 .- pR_n
        p_sys_NLEL_NRER[:,:,:,1,1] *= prod(pR_n0)

        # NR>=1
        for jjNR = 1:K
            labelR = collect(combinations(1:K,jjNR))
            for jjNlabelR = 1:length(labelR)

                pR_n0 .= 1.0 .- pR_n
                pR_n0[labelR[jjNlabelR]] = pR_n[labelR[jjNlabelR]]
                p_sys_NLEL_NRER[:,:,:,jjNR+1,jjNlabelR] *= prod(pR_n0)

                for jjdeg = 1:ind
                    if jjNR == pLR_NE_deg[jjdeg,1] && jjNlabelR == pLR_NE_deg[jjdeg,2]
                       p_sys_NLEL_NRER[:,:,:,jjNR+1,pLR_NE_deg[jjdeg,3]] += p_sys_NLEL_NRER[:,:,:,jjNR+1,jjNlabelR]
                       p_sys_NLEL_NRER[:,:,:,jjNR+1,jjNlabelR] .= 0.0
                       break #
                    end
                end
            end
        end

        # p_NLEL_NRER
        p_NLEL_NRER .= p_sys_NLEL_NRER[1,:,:,:,:]/(1-psys_n)

        # observational entropy and relative entropy
        for jjN = 1:K+1
            for jjNlabel = 1:binomial(K,jjN-1)

                if pLR_NE_V[jjN,jjNlabel] > 0
                   SobsL[tt] += pL_NE[jjN,jjNlabel]*(-log(pL_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   SobsR[tt] += pR_NE[jjN,jjNlabel]*(-log(pR_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   DrelL_obs[tt] += pL_NE[jjN,jjNlabel]*(log(pL_NE[jjN,jjNlabel])-log(pL_NE_t0[jjN,jjNlabel]))
                   DrelR_obs[tt] += pR_NE[jjN,jjNlabel]*(log(pR_NE[jjN,jjNlabel])-log(pR_NE_t0[jjN,jjNlabel]))
                end

            end
        end


        for jjNL = 1:K+1
            for jjNlabelL = 1:binomial(K,jjNL-1)
                if pLR_NE_V[jjNL,jjNlabelL] <= 0
                   continue
                end
                for jjNR = 1:K+1
                    for jjNlabelR = 1:binomial(K,jjNR-1)
                        if pLR_NE_V[jjNR,jjNlabelR] <= 0
                           continue
                        end
                        SobsE[tt] += p_NLEL_NRER[jjNL,jjNlabelL,jjNR,jjNlabelR]*(-log(p_NLEL_NRER[jjNL,jjNlabelL,jjNR,jjNlabelR])+log(pLR_NE_V[jjNL,jjNlabelL]*pLR_NE_V[jjNR,jjNlabelR]))
                        for jjNsys = 1:2
                            Sobs[tt] += p_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR]*(-log(p_sys_NLEL_NRER[jjNsys,jjNL,jjNlabelL,jjNR,jjNlabelR])+log(pLR_NE_V[jjNL,jjNlabelL]*pLR_NE_V[jjNR,jjNlabelR]))
                        end
                    end
                end
            end
        end

        # system
        Sobssys[tt] = real(-Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1]))

        # correlation
        Iobs_SE[tt] = Sobssys[tt] - Sobssys[1] + SobsE[tt] - SobsE[1] - (Sobs[tt] - Sobs[1])
        Iobs_B[tt] = SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] - (SobsE[tt] - SobsE[1])

        # entropy production of obs
        sigmaobs[tt] = Sobssys[tt] - Sobssys[1] + SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] + DrelL_obs[tt] + DrelR_obs[tt]

        # entropy production of vNE
        vNE_L[tt] = real(- sum(eigval_Ct_L.*log.(eigval_Ct_L)) - sum((1.0 .- eigval_Ct_L).*log.(1.0 .- eigval_Ct_L)))
        vNE_R[tt] = real(- sum(eigval_Ct_R.*log.(eigval_Ct_R)) - sum((1.0 .- eigval_Ct_R).*log.(1.0 .- eigval_Ct_R)))
        vNE[tt] = real(- sum(eigval_Ct.*log.(eigval_Ct)) - sum((1.0 .- eigval_Ct).*log.(1.0 .- eigval_Ct)))
        eigval_Ct_E = eigvals(Ct[2:end,2:end])
        vNE_E[tt] = real(- sum(eigval_Ct_E.*log.(eigval_Ct_E)) - sum((1.0 .- eigval_Ct_E).*log.(1.0 .- eigval_Ct_E)))

        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        betaQR[tt] = QR[tt]*betaR
        sigma[tt] = Sobssys[tt] - Sobssys[1] - betaQL[tt] - betaQR[tt]
        Drel_vNE[tt] = sigma[tt] - (Sobssys[tt] - Sobssys[1] + vNE_L[tt] - vNE_L[1] + vNE_R[tt] - vNE_R[1])

    end

    # return p_sys_NLEL_NRER

    # return time, Sobs, Sobssys, SobsE, SobsL, SobsR, DrelL_obs, DrelR_obs
    return time, sigmaobs, Sobs, Iobs_SE, Iobs_B, DrelL_obs, DrelR_obs, sigma
    # return time, SobsL, SobsR, Sobs, SobsE, vNE_L, vNE_R, vNE, vNE_E
    # return time, SobsL, SobsR, vNE_L, vNE_R, Sobssys, sigmaobs, sigma, DrelL_obs, DrelR_obs, Drel_vNE

end

function calculate_Sobs_test(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2 or small
    # K = 8

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_fluctuatedt!(K,W,t_flu,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH))
    tLRk = matH[1,1:end]

    epsilonL = epsilonLR[2:K+1]
    epsilonR = epsilonLR[K+2:2*K+1]

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)
    # eigval_Ct = zeros(Float64,2*K+1)
    # eigvec_Ct = zeros(Float64,2*K+1,2*K+1)
    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    pL_n = zeros(Float64,K)
    pL_n0 = zeros(Float64,K)
    pR_n = zeros(Float64,K)
    pR_n0 = zeros(Float64,K)

    pL_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pL_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    pR_NE = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))
    pR_NE_t0 = zeros(Float64,K+1,binomial(K,floor(Int64,K/2)))

    pLR_NE_E = ones(Int64,K+1,binomial(K,floor(Int64,K/2)))*(-1)
    pLR_NE_V = zeros(Int64,K+1,binomial(K,floor(Int64,K/2)))
    pLR_NE_deg = zeros(Int64,2^K,3)

    SobsL = zeros(Float64,Nt)
    SobsR = zeros(Float64,Nt)
    DrelL_obs = zeros(Float64,Nt)
    DrelR_obs = zeros(Float64,Nt)
    Sobssys = zeros(Float64,Nt)
    sigmaobs = zeros(Float64,Nt)

    dCt = zeros(ComplexF64,K*2+1)
    QL = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    Drel_vNE = zeros(ComplexF64,Nt)

    pL_t = zeros(Float64,K,Nt)
    # p_t = zeros(Float64,2*K+1,Nt)

    # pL_EN_E,V for N=0 and so E=0
    pLR_NE_E[1,1] = 0
    pLR_NE_V[1,1] = 1
    ind = 0

    # pLR_NE_E,V for N>=1
    for jjN = 1:K
        label = collect(combinations(1:K,jjN))
        for jjNlabel = 1:length(label)

            pLR_NE_E[1+jjN,jjNlabel] = sum(label[jjNlabel])
            pLR_NE_V[1+jjN,jjNlabel] = 1

            # check degenracy
            for jjcheck = 1:jjNlabel-1
                if pLR_NE_E[1+jjN,jjcheck] == pLR_NE_E[1+jjN,jjNlabel] #&& pL_NE_V[1+jjN,jjcheck] > 0
                   ind += 1
                   pLR_NE_deg[ind,1] = jjN
                   pLR_NE_deg[ind,2] = jjNlabel
                   pLR_NE_deg[ind,3] = jjcheck
                   pLR_NE_V[1+jjN,jjcheck] += 1
                   pLR_NE_V[1+jjN,jjNlabel] = -1
                   break
                end
            end

        end
    end
    pLR_NE_deg = pLR_NE_deg[1:ind,:]

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # total
        # lambda, eigvec_Ct = eigen(Ct)
        # eigval_Ct .= real.(lambda)

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)
        pL_n .= (abs.(eigvec_Ct_L).^2)*eigval_Ct_L
        # pL_n .= diag(Ct[2:K+1,2:K+1])

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)
        pR_n .= (abs.(eigvec_Ct_R).^2)*eigval_Ct_R
        # pR_n .= diag(Ct[K+2:2*K+1,K+2:2*K+1])

        # for jj = 1:K
        #     if pL_n[jj] < 0.0001
        #        pL_n[jj] = 0.0+10^(-15)
        #     elseif pL_n[jj] > 0.9999
        #        pL_n[jj] = 1.0-10^(-15)
        #     end
        # end
        # pL_t[:,tt] = pL_n
        # pL_t[:,tt] = pL_n - real(diag(Ct[2:K+1,2:K+1]))

        pL_NE .= 0.0
        pR_NE .= 0.0

        # pL_EN for N=0 and so E=0
        pL_n0 .= 1.0 .- pL_n
        pL_NE[1,1] = prod(pL_n0)

        # pR_EN for N=0 and so E=0
        pR_n0 .= 1.0 .- pR_n
        pR_NE[1,1] = prod(pR_n0)

        # # only for K=2
        # pL_NE[2,1] = pL_n[1]*(1-pL_n[2])
        # pL_NE[2,2] = (1-pL_n[1])*pL_n[2]
        # pL_NE[3,1] = pL_n[1]*pL_n[2]

        # # only for K=4
        # pR_NE[2,1] = pR_n[1]*(1-pR_n[2])*(1-pR_n[3])*(1-pR_n[4])
        # pR_NE[2,2] = (1-pR_n[1])*pR_n[2]*(1-pR_n[3])*(1-pR_n[4])
        # pR_NE[2,3] = (1-pR_n[1])*(1-pR_n[2])*pR_n[3]*(1-pR_n[4])
        # pR_NE[2,4] = (1-pR_n[1])*(1-pR_n[2])*(1-pR_n[3])*pR_n[4]
        # pR_NE[3,1] = pR_n[1]*pR_n[2]*(1-pR_n[3])*(1-pR_n[4])
        # pR_NE[3,2] = pR_n[1]*(1-pR_n[2])*pR_n[3]*(1-pR_n[4])
        # pR_NE[3,3] = pR_n[1]*(1-pR_n[2])*(1-pR_n[3])*pR_n[4]
        # pR_NE[3,4] = (1-pR_n[1])*pR_n[2]*pR_n[3]*(1-pR_n[4])
        # pR_NE[3,5] = (1-pR_n[1])*pR_n[2]*(1-pR_n[3])*pR_n[4]
        # pR_NE[3,6] = (1-pR_n[1])*(1-pR_n[2])*pR_n[3]*pR_n[4]
        # pR_NE[4,1] = pR_n[1]*pR_n[2]*pR_n[3]*(1-pR_n[4])
        # pR_NE[4,2] = pR_n[1]*pR_n[2]*(1-pR_n[3])*pR_n[4]
        # pR_NE[4,3] = pR_n[1]*(1-pR_n[2])*pR_n[3]*pR_n[4]
        # pR_NE[4,4] = (1-pR_n[1])*pR_n[2]*pR_n[3]*pR_n[4]
        # pR_NE[5,1] = pR_n[1]*pR_n[2]*pR_n[3]*pR_n[4]
        # pR_NE[3,3] += pR_NE[3,4]
        # pR_NE[3,4] = 0

        # pL_EN and pR_EN for N>=1
        for jjN = 1:K
            label = collect(combinations(1:K,jjN))
            for jjNlabel = 1:length(label)

                # L
                pL_n0 .= 1.0 .- pL_n
                pL_n0[label[jjNlabel]] = pL_n[label[jjNlabel]]
                pL_NE[1+jjN,jjNlabel] = prod(pL_n0)

                # R
                pR_n0 .= 1.0 .- pR_n
                pR_n0[label[jjNlabel]] = pR_n[label[jjNlabel]]
                pR_NE[1+jjN,jjNlabel] = prod(pR_n0)

                for jjdeg = 1:ind
                    if jjN == pLR_NE_deg[jjdeg,1] && jjNlabel == pLR_NE_deg[jjdeg,2]
                       pL_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pL_NE[1+jjN,jjNlabel]
                       pL_NE[1+jjN,jjNlabel] = 0.0
                       pR_NE[1+jjN,pLR_NE_deg[jjdeg,3]] += pR_NE[1+jjN,jjNlabel]
                       pR_NE[1+jjN,jjNlabel] = 0.0

                       #
                       break

                    end
                end

            end
        end
        if tt == 1
           pL_NE_t0 .= pL_NE
           pR_NE_t0 .= pR_NE
        end

        # return pLR_NE_V, pLR_NE_deg, pL_NE

        # observational entropy and relative entropy
        for jjN = 1:K+1
            for jjNlabel = 1:binomial(K,jjN-1)

                if pLR_NE_V[jjN,jjNlabel] > 0
                   SobsL[tt] += pL_NE[jjN,jjNlabel]*(-log(pL_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   SobsR[tt] += pR_NE[jjN,jjNlabel]*(-log(pR_NE[jjN,jjNlabel])+log(pLR_NE_V[jjN,jjNlabel]))
                   DrelL_obs[tt] += pL_NE[jjN,jjNlabel]*(log(pL_NE[jjN,jjNlabel])-log(pL_NE_t0[jjN,jjNlabel]))
                   DrelR_obs[tt] += pR_NE[jjN,jjNlabel]*(log(pR_NE[jjN,jjNlabel])-log(pR_NE_t0[jjN,jjNlabel]))
                end

            end
        end

        # system
        Sobssys[tt] = real(-Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1]))

        # entropy production of obs
        sigmaobs[tt] = Sobssys[tt] - Sobssys[1] + SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] + DrelL_obs[tt] + DrelR_obs[tt]

        # entropy production of vNE
        vNE_L[tt] = real(- sum(eigval_Ct_L.*log.(eigval_Ct_L)) - sum((1.0 .- eigval_Ct_L).*log.(1.0 .- eigval_Ct_L)))
        vNE_R[tt] = real(- sum(eigval_Ct_R.*log.(eigval_Ct_R)) - sum((1.0 .- eigval_Ct_R).*log.(1.0 .- eigval_Ct_R)))
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:K+1].*(epsilonLR[2:K+1] .- muL))
        betaQL[tt] = QL[tt]*betaL
        QR[tt] = -sum(dCt[K+2:2*K+1].*(epsilonLR[K+2:2*K+1] .- muR))
        betaQR[tt] = QR[tt]*betaR
        sigma[tt] = Sobssys[tt] - Sobssys[1] - betaQL[tt] - betaQR[tt]
        Drel_vNE[tt] = sigma[tt] - (Sobssys[tt] - Sobssys[1] + vNE_L[tt] - vNE_L[1] + vNE_R[tt] - vNE_R[1])

    end

    return pL_t

    # return time, SobsL, SobsR, vNE_L, vNE_R, pL_NE, pR_NE, pLR_NE_E, pLR_NE_V, pLR_NE_deg
    # return time, SobsL, SobsR
    return time, SobsL, SobsR, vNE_L, vNE_R, Sobssys, sigmaobs, sigma, DrelL_obs, DrelR_obs, Drel_vNE
    # return time, Sobssys, vNE_L, vNE_R, vNE_total
    # return time, SobsL, SobsR, vNE_L, vNE_R

end

function calculate_Sobs_K2(W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2
    K = 2

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end

    epsilon = diag(matH)
    Ct = zeros(ComplexF64,2*K+1,2*K+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)
    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    ptotal = zeros(Float64,2^(2*K+1))
    ntotalj = zeros(Float64,2*K+1)

    pL = zeros(Float64,2^K)
    nLj = ntotalj = zeros(Float64,K)
    pL0 = zeros(Float64,2^K)

    pR = zeros(Float64,2^K)
    nRj = ntotalj = zeros(Float64,K)
    pR0 = zeros(Float64,2^K)

    Sobs = zeros(Float64,Nt)
    SobsL = zeros(Float64,Nt)
    SobsR = zeros(Float64,Nt)
    Drel = zeros(Float64,Nt)
    Sobssys = zeros(Float64,Nt)
    sigmaobs = zeros(Float64,Nt)

    diagC0 = C0
    C0 = diagm(C0)

    # vNE of the total system
    # vNE = - sum(diagC0.*log.(diagC0)) - sum((1.0 .- diagC0).*log.(1.0 .- diagC0))
    # println("vNE")
    # println(vNE)

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # total
        lambda, eigvec_Ct = eigen(Ct)
        eigval_Ct .= real.(lambda)
        ntotalj = (abs.(eigvec_Ct).^2)*eigval_Ct
        # ntotalj[:,1] .= (abs.(eigvec_Ct).^2)*eigval_Ct
        # ntotalj[:,2] .= 1 .- ntotalj[:,1]

        ptotal .= 0.0
        ptotal[1] = ntotalj[1]*ntotalj[2]*ntotalj[3]*ntotalj[4]*ntotalj[5]
        ptotal[2] = ntotalj[1]*ntotalj[2]*ntotalj[3]*ntotalj[4]*(1-ntotalj[5])

        ptotal[3] = ntotalj[1]*ntotalj[2]*ntotalj[3]*(1-ntotalj[4])*ntotalj[5]
        ptotal[4] = ntotalj[1]*ntotalj[2]*ntotalj[3]*(1-ntotalj[4])*(1-ntotalj[5])

        ptotal[5] = ntotalj[1]*ntotalj[2]*(1-ntotalj[3])*ntotalj[4]*ntotalj[5]
        ptotal[6] = ntotalj[1]*ntotalj[2]*(1-ntotalj[3])*ntotalj[4]*(1-ntotalj[5])
        ptotal[7] = ntotalj[1]*ntotalj[2]*(1-ntotalj[3])*(1-ntotalj[4])*ntotalj[5]
        ptotal[8] = ntotalj[1]*ntotalj[2]*(1-ntotalj[3])*(1-ntotalj[4])*(1-ntotalj[5])

        ptotal[9] = ntotalj[1]*(1-ntotalj[2])*ntotalj[3]*ntotalj[4]*ntotalj[5]
        ptotal[10] = ntotalj[1]*(1-ntotalj[2])*ntotalj[3]*ntotalj[4]*(1-ntotalj[5])
        ptotal[11] = ntotalj[1]*(1-ntotalj[2])*ntotalj[3]*(1-ntotalj[4])*ntotalj[5]
        ptotal[12] = ntotalj[1]*(1-ntotalj[2])*ntotalj[3]*(1-ntotalj[4])*(1-ntotalj[5])
        ptotal[13] = ntotalj[1]*(1-ntotalj[2])*(1-ntotalj[3])*ntotalj[4]*ntotalj[5]
        ptotal[14] = ntotalj[1]*(1-ntotalj[2])*(1-ntotalj[3])*ntotalj[4]*(1-ntotalj[5])
        ptotal[15] = ntotalj[1]*(1-ntotalj[2])*(1-ntotalj[3])*(1-ntotalj[4])*ntotalj[5]
        ptotal[16] = ntotalj[1]*(1-ntotalj[2])*(1-ntotalj[3])*(1-ntotalj[4])*(1-ntotalj[5])

        ptotal[17] = (1-ntotalj[1])*ntotalj[2]*ntotalj[3]*ntotalj[4]*ntotalj[5]
        ptotal[18] = (1-ntotalj[1])*ntotalj[2]*ntotalj[3]*ntotalj[4]*(1-ntotalj[5])
        ptotal[19] = (1-ntotalj[1])*ntotalj[2]*ntotalj[3]*(1-ntotalj[4])*ntotalj[5]
        ptotal[20] = (1-ntotalj[1])*ntotalj[2]*ntotalj[3]*(1-ntotalj[4])*(1-ntotalj[5])
        ptotal[21] = (1-ntotalj[1])*ntotalj[2]*(1-ntotalj[3])*ntotalj[4]*ntotalj[5]
        ptotal[22] = (1-ntotalj[1])*ntotalj[2]*(1-ntotalj[3])*ntotalj[4]*(1-ntotalj[5])
        ptotal[23] = (1-ntotalj[1])*ntotalj[2]*(1-ntotalj[3])*(1-ntotalj[4])*ntotalj[5]
        ptotal[24] = (1-ntotalj[1])*ntotalj[2]*(1-ntotalj[3])*(1-ntotalj[4])*(1-ntotalj[5])
        ptotal[25] = (1-ntotalj[1])*(1-ntotalj[2])*ntotalj[3]*ntotalj[4]*ntotalj[5]
        ptotal[26] = (1-ntotalj[1])*(1-ntotalj[2])*ntotalj[3]*ntotalj[4]*(1-ntotalj[5])
        ptotal[27] = (1-ntotalj[1])*(1-ntotalj[2])*ntotalj[3]*(1-ntotalj[4])*ntotalj[5]
        ptotal[28] = (1-ntotalj[1])*(1-ntotalj[2])*ntotalj[3]*(1-ntotalj[4])*(1-ntotalj[5])
        ptotal[29] = (1-ntotalj[1])*(1-ntotalj[2])*(1-ntotalj[3])*ntotalj[4]*ntotalj[5]
        ptotal[30] = (1-ntotalj[1])*(1-ntotalj[2])*(1-ntotalj[3])*ntotalj[4]*(1-ntotalj[5])
        ptotal[31] = (1-ntotalj[1])*(1-ntotalj[2])*(1-ntotalj[3])*(1-ntotalj[4])*ntotalj[5]
        ptotal[32] = (1-ntotalj[1])*(1-ntotalj[2])*(1-ntotalj[3])*(1-ntotalj[4])*(1-ntotalj[5])
        Sobs[tt] = -sum(ptotal.*log.(ptotal))

        # L
        pL .= 0.0
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)
        nLj .= (abs.(eigvec_Ct_L).^2)*eigval_Ct_L
        pL[1] = nLj[1]*nLj[2]
        pL[2] = nLj[1]*(1-nLj[2])
        pL[3] = (1-nLj[1])*nLj[2]
        pL[4] = (1-nLj[1])*(1-nLj[2])
        SobsL[tt] = -sum(pL.*log.(pL))
        if tt == 1
           pL0 .= pL
        end

        # R
        pR .= 0.0
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)
        nRj .= (abs.(eigvec_Ct_R).^2)*eigval_Ct_R
        pR[1] = nRj[1]*nRj[2]
        pR[2] = nRj[1]*(1-nRj[2])
        pR[3] = (1-nRj[1])*nRj[2]
        pR[4] = (1-nRj[1])*(1-nRj[2])
        SobsR[tt] = -sum(pR.*log.(pR))
        if tt == 1
           pR0 .= pR
        end

        # relative entropy
        Drel[tt] = sum(pL.*(log.(pL) .- log.(pL0))) + sum(pR.*(log.(pR) .- log.(pR0)))

        # system
        Sobssys[tt] = real(-Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1]))

        # entropy production
        sigmaobs[tt] = Sobssys[tt] - Sobssys[1] + SobsL[tt] - SobsL[1] + SobsR[tt] - SobsR[1] + Drel[tt]

    end

    return time, Sobs, SobsL, SobsR, Sobssys, Drel, sigmaobs

end

################ basis functions for population distribution in E_j and N_j

function roundeigval_Ct!(eigval_Ct::Vector{Float64},p_part::Vector{Float64},count::Vector{Int64},counteps_1::Vector{Int64},criterion::Float64)

    # count the number of Tr[n_j rho] close to 1 or 0
    p_part .= 0.0
    count .= 0
    count_1 = 0
    # count_0 = 0
    ind = 0
    counteps_1 .= 0

    for jj = 1:length(eigval_Ct)
        if eigval_Ct[jj] > 1.0 - criterion
           p_part[jj] = 1.0
           count_1 += 1
           counteps_1[count_1] = jj
        elseif eigval_Ct[jj] < criterion
           p_part[jj] = 1.0 - 0.0
           # count_0 += 1
        else
           p_part[jj] = 1.0 - eigval_Ct[jj]
           ind += 1
           count[ind] = jj
        end
    end

    return ind, count_1

end

function popdistri_degenreate!(p::Matrix{Float64},arrayE::Vector{Float64},counteps_1::Vector{Int64},count_1::Int64,p_part::Vector{Float64},epsilon_tilde::Vector{Float64},ind::Int64,count::Vector{Int64},p_part_comb::Vector{Float64})

    p .= 0.0
    arrayE .= 0.0
    indE = 0
    arrayE0 = 0.0

    indE += 1
    p[1+count_1,indE] = prod(p_part[:])
    arrayE0 = sum(epsilon_tilde[counteps_1[1:count_1]])
    arrayE[indE] = arrayE0

    for jjN = 1:ind

        combind = collect(combinations(count[1:ind],jjN))
        Mcombind = length(combind)

        for iiN = 1:Mcombind
            p_part_comb .= p_part
            for kkN = 1:jjN
                p_part_comb[combind[iiN][kkN]] = 1.0-p_part_comb[combind[iiN][kkN]] #eigval_Ct[combind[iiN][kkN]]
            end
            indE += 1
            p[1+count_1+jjN,indE] += prod(p_part_comb[:])
            arrayE[indE] = sum(epsilon_tilde[combind[iiN]])+arrayE0
        end

    end

    # sort arrayE as well as others
    indarrayE = sortperm(arrayE[1:indE])
    arrayE[1:indE] = arrayE[indarrayE]
    for jjN = 0:ind
        p[1+count_1+jjN,1:indE] = p[1+count_1+jjN,indarrayE]
    end

    return indE, [count_1, count_1+ind]

end

function roundarrayE!(arrayE::Vector{Float64},indE::Int64,arrayEround::Vector{Float64},ind::Int64,pround::Matrix{Float64},count_1::Int64,p::Matrix{Float64})

    check0 = arrayE[1]
    indcheck0 = 0
    indround = 0
    for jjE = 2:indE

        if abs(arrayE[jjE]-check0) < 10^(-10)
           indcheck0 += 1
        else
           indround += 1
           arrayEround[indround] = check0

           for jjN = 0:ind
               pround[1+count_1+jjN,indround] = sum(p[1+count_1+jjN,jjE-1-indcheck0:jjE-1])
           end
           check0 = arrayE[jjE]
           indcheck0 = 0
        end

    end
    indround += 1
    arrayEround[indround] = check0
    for jjN = 0:ind
        pround[1+count_1+jjN,indround] = sum(p[1+count_1+jjN,indE-indcheck0:indE])
    end

    return indround, arrayEround, pround

end

function obsentropy(pround::Matrix{Float64},ind::Int64,indround::Int64,count_1::Int64)

    Sobs = 0.0

    for jjN = 0:ind
        for jjE = 1:indround
            pop = pround[1+count_1+jjN,jjE]
            if pop != 0.0
               Sobs += -pop*log(pop)
            end
        end
    end

    return Sobs

end

function relentropy(pround::Matrix{Float64},jjN01tt::Vector{Int64},jjEendtt::Int64,arrayEroundtt::Vector{Float64},state::Matrix{Float64},jjN01::Vector{Int64},jjEend::Int64,arrayEround::Vector{Float64})

    Drel = 0.0

    for jjN = jjN01tt[1]:jjN01tt[2]
        for jjE = 1:indround
            pop = pround[jjN,jjE]
            popstate = state[1+count_1+jjN,jjE]
            if pop != 0.0 && popstate != 0.0
               Drel += pop*(log(pop)-log(popstate))
            end
        end
    end

    return Drel

end

function calculateptotal_test3(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilon_tilde = zeros(Float64,2*K+1,Nt)
    indtilde = zeros(Int64,2*K+1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)

    # Nenebath = Int64(K*(K+1)/2)
    lengthErange = 2^22 #2^21
    ptotal = zeros(Float64,2*K+1+1,lengthErange)
    ptotalround = zeros(Float64,2*K+1+1,lengthErange,Nt)
    ptotal_part = zeros(Float64,2*K+1)
    ptotal_part_comb = zeros(Float64,2*K+1)
    criterion = 0.0

    count_total = zeros(Int64,2*K+1)
    count_total1 = 0
    count_total0 = 0
    counteps_total1 = zeros(Int64,2*K+1)
    ind = 0

    indE = 0
    arrayE = zeros(Float64,lengthErange) #spzeros(Float64,lengthErange)
    arrayEround = zeros(Float64,lengthErange,Nt)
    arrayEroundsize = zeros(Int64,Nt)
    arrayEsize = zeros(Int64,Nt)
    arrayN = zeros(Int64,2,Nt)
    indround = 0

    Sobstotal = zeros(Float64,Nt)

    for tt = 1:Nt

        println("t=",tt)

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        #
        lambda, eigvec_Ct = eigen(Ct)
        eigval_Ct .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:2*K+1
            epsilon_tilde[ss,tt] = sum(abs.(eigvec_Ct[:,ss]).^2 .* epsilon)
        end

        # count the number of Tr[n_j rho] close to 1 or 0
        ind, count_total1 = roundeigval_Ct!(eigval_Ct,ptotal_part,count_total,counteps_total1,criterion)

        # construct population distribution
        indE, arrayN[:,tt] = popdistri_degenreate!(ptotal,arrayE,counteps_total1,count_total1,ptotal_part,epsilon_tilde[:,tt],ind,count_total,ptotal_part_comb)

        # round E
        arrayEroundsize[tt], arrayEround[:,tt], ptotalround[:,:,tt] = roundarrayE!(arrayE,indE,arrayEround[:,tt],ind,ptotalround[:,:,tt],count_total1,ptotal)

        # Sobs
        Sobstotal[tt] = obsentropy(ptotalround[:,:,tt],ind,arrayEroundsize[tt],count_total1)

    end

    return time, arrayEround, arrayEroundsize, arrayN, ptotalround, Sobstotal

end

function calculatepLR_test5(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilonL_tilde = zeros(Float64,K,Nt)
    epsilonR_tilde = zeros(Float64,K,Nt)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)

    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    lengthErange = 2^22 #2^21
    pL = zeros(Float64,K+1,lengthErange) # N_j counts from N_j=0 and so the index is N_j+1
    pLround = zeros(Float64,K+1,lengthErange,Nt)
    pL_part = zeros(Float64,K)
    pL_part_comb = zeros(Float64,K)
    pR = zeros(Float64,K+1,lengthErange)
    pRround = zeros(Float64,K+1,lengthErange,Nt)
    pR_part = zeros(Float64,K)
    pR_part_comb = zeros(Float64,K)
    criterion = 0.0

    count_L = zeros(Int64,K)
    counteps_L1 = zeros(Int64,K)
    count_R = zeros(Int64,K)
    counteps_R1 = zeros(Int64,K)

    arrayEL = zeros(Float64,lengthErange) #spzeros(Float64,lengthErange)
    arrayELround = zeros(Float64,lengthErange,Nt)
    arrayELroundsize = zeros(Int64,Nt)
    arrayELsize = zeros(Int64,Nt)
    arrayNL = zeros(Int64,2,Nt)
    indLround = 0
    arrayER = zeros(Float64,lengthErange)
    arrayERround = zeros(Float64,lengthErange,Nt)
    arrayERroundsize = zeros(Int64,Nt)
    arrayERsize = zeros(Int64,Nt)
    arrayNR = zeros(Int64,2,Nt)
    indRround = 0

    SobsL = zeros(Float64,Nt)
    SobsR = zeros(Float64,Nt)

    # stateL = zeros(Float64,K+1,lengthErange)
    # DrelL = zeros(Float64,Nt)
    # stateR = zeros(Float64,K+1,lengthErange)
    # DrelR = zeros(Float64,Nt)

    for tt = 1:Nt

        println("t=",tt)

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:K
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilonL)
            epsilonR_tilde[ss,tt] = sum(abs.(eigvec_Ct_R[:,ss]).^2 .* epsilonR)
        end

        # count the number of Tr[n_j rho] close to 1 or 0
        indL, count_L1 = roundeigval_Ct!(eigval_Ct_L,pL_part,count_L,counteps_L1,criterion)
        indR, count_R1 = roundeigval_Ct!(eigval_Ct_R,pR_part,count_R,counteps_R1,criterion)

        # construct population distribution
        indEL, arrayNL[:,tt] = popdistri_degenreate!(pL,arrayEL,counteps_L1,count_L1,pL_part,epsilonL_tilde[:,tt],indL,count_L,pL_part_comb)
        indER, arrayNR[:,tt] = popdistri_degenreate!(pR,arrayER,counteps_R1,count_R1,pR_part,epsilonR_tilde[:,tt],indR,count_R,pR_part_comb)

        # round E
        arrayELroundsize[tt], arrayELround[:,tt], pLround[:,:,tt] = roundarrayE!(arrayEL,indEL,arrayELround[:,tt],indL,pLround[:,:,tt],count_L1,pL)
        arrayERroundsize[tt], arrayERround[:,tt], pRround[:,:,tt] = roundarrayE!(arrayER,indER,arrayERround[:,tt],indR,pRround[:,:,tt],count_R1,pR)

        # Sobs
        SobsL[tt] = obsentropy(pLround[:,:,tt],indL,arrayELroundsize[tt],count_L1)
        SobsR[tt] = obsentropy(pRround[:,:,tt],indR,arrayERroundsize[tt],count_R1)

        # Drel
        # DrelL[tt] = relentropy(pLround[:,:,tt],arrayNL[:,tt].+1,arrayELroundsize[tt],arrayELround[:,tt],stateL,arrayNL[:,1].+1,arrayELroundsize[1],arrayELround[:,1])
        # DrelR[tt] = relentropy(pRround[:,:,tt],arrayNR[:,tt].+1,arrayERroundsize[tt],arrayERround[:,tt],stateR,arrayNR[:,1].+1,arrayERroundsize[1],arrayERround[:,1])

    end

    return time, arrayELround, arrayERround, arrayELroundsize, arrayERroundsize, arrayNL, arrayNR, pLround, pRround, SobsL, SobsR

end

function calculateSigmaobs_test(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilon_tilde = zeros(Float64,2*K+1,Nt)
    indtilde = zeros(Int64,2*K+1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)

    # Nenebath = Int64(K*(K+1)/2)
    lengthErange = 2^22 #2^21
    ptotal = zeros(Float64,2*K+1+1,lengthErange)
    ptotalround = zeros(Float64,2*K+1+1,lengthErange,Nt)
    ptotal_part = zeros(Float64,2*K+1)
    ptotal_part_comb = zeros(Float64,2*K+1)
    criterion = 0.00001

    count_total = zeros(Int64,2*K+1)
    count_total1 = 0
    count_total0 = 0
    counteps_total1 = zeros(Int64,2*K+1)
    ind = 0

    indE = 0
    arrayE = zeros(Float64,lengthErange) #spzeros(Float64,lengthErange)
    arrayEround = zeros(Float64,lengthErange,Nt)
    arrayEroundsize = zeros(Int64,Nt)
    arrayEsize = zeros(Int64,Nt)
    arrayN = zeros(Int64,2,Nt)

    for tt = 1:Nt

        println("t=",tt)

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # bath L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # bath R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:2*K+1
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilon)
            epsilonR_tilde[ss,tt] = sum(abs.(eigvec_Ct_R[:,ss]).^2 .* epsilon)
        end

        # count the number of Tr[n_j rho] close to 1 or 0
        roundeigval_Ct!(eigval_Ct_L,pL_part,count_L,count_L1,count_L0,indL,counteps_L1,K,criterion)
        roundeigval_Ct!(eigval_Ct_R,pR_part,count_R,count_R1,count_R0,indR,counteps_R1,K,criterion)

        # construct population distribution
        popdistri_degenreate!(pL,arrayEL,indEL,arrayEL0,count_L1,pL_part,epsilonL_tilde[:,tt],indL,pL_part_comb)
        arrayNL[1,tt] = count_L1
        arrayNL[2,tt] = count_L1+indL
        popdistri_degenreate!(pR,arrayER,indER,arrayER0,count_R1,pR_part,epsilonR_tilde[:,tt],indR,pR_part_comb)
        arrayNR[1,tt] = count_R1
        arrayNR[2,tt] = count_R1+indR

        # round E
        roundarrayE!(arrayEL,indEL,arrayELround,indL,pLround,count_L1,pL,indLround)
        arrayELroundsize[tt] = indLround
        roundarrayE!(arrayER,indER,arrayERround,indR,pRround,count_R1,pR,indRround)
        arrayERroundsize[tt] = indRround

    end

    return time, arrayEround, arrayEroundsize, arrayN, ptotalround

end

function calculateptotal_test2(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    # C0 = diagm(C0)
    # epsilon = Array(diag(matH))
    arrayE = [-W, -W/2, 0.0, W/2, W]
    ptotal = zeros(Float64,5,5)

    println(arrayE)
    println(C0)

    ptotal[5,3] += C0[1]*C0[2]*C0[3]*C0[4]*C0[5]
    ptotal[4,2] += C0[1]*C0[2]*C0[3]*C0[4]*(1-C0[5])

    ptotal[4,4] += C0[1]*C0[2]*C0[3]*(1-C0[4])*C0[5]
    ptotal[3,3] += C0[1]*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])

    ptotal[4,2] += C0[1]*C0[2]*(1-C0[3])*C0[4]*C0[5]
    ptotal[3,1] += C0[1]*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])
    ptotal[3,3] += C0[1]*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]
    ptotal[2,2] += C0[1]*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])

    ptotal[4,4] += C0[1]*(1-C0[2])*C0[3]*C0[4]*C0[5]
    ptotal[3,3] += C0[1]*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])
    ptotal[3,5] += C0[1]*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]
    ptotal[2,4] += C0[1]*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])
    ptotal[3,3] += C0[1]*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]
    ptotal[2,2] += C0[1]*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])
    ptotal[2,4] += C0[1]*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]
    ptotal[1,3] += C0[1]*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])

    ptotal[4,3] += (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]
    ptotal[3,2] += (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])
    ptotal[3,4] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]
    ptotal[2,3] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])
    ptotal[3,2] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]
    ptotal[2,1] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])
    ptotal[2,3] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]
    ptotal[1,2] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])
    ptotal[3,4] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]
    ptotal[2,3] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])
    ptotal[2,5] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]
    ptotal[1,4] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])
    ptotal[2,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]
    ptotal[1,2] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])
    ptotal[1,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]
    # ptotal[0,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])

    ptotal0 = (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])
    test0 = zeros(Float64,1,5)
    test0[1,3] = ptotal0
    ptotal = [test0;ptotal]
    # ptotal = ptotal .+ 1e-15

    Sobs = 0.0
    for jj1 = 1:6
        for jj2 = 1:5
            pop = ptotal[jj1,jj2]
            if abs(pop) != 0.0
               Sobs += -pop*log(pop)
            end
        end
    end

    # vNE
    vNE = - sum(C0.*log.(C0)) - sum((1.0 .- C0).*log.(1.0 .- C0))

    println(vNE)

    # return ptotal
    return Sobs

end

function calculateptotal_test4(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=3

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    # C0 = diagm(C0)
    epsilon = Array(diag(matH))
    arrayE = [-W,-W/2,0.0,W/2,W] #[epsilon[2]+epsilon[4],epsilon[2],epsilon[1],epsilon[3],epsilon[3]+epsilon[5]]
    ptotal = zeros(Float64,7,5)

    println(arrayE)
    println(C0)

    ptotal[6,3] += (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]*C0[6]*C0[7]
    ptotal[5,2] += (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]*C0[6]*(1-C0[7])

    ptotal[5,3] += (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]*(1-C0[6])*C0[7]
    ptotal[4,2] += (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]*(1-C0[6])*(1-C0[7])

    ptotal[5,4] += (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])*C0[6]*C0[7]
    ptotal[4,3] += (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[4,4] += (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[3,3] += (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])*(1-C0[6])*(1-C0[7])

    ptotal[5,2] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]*C0[6]*C0[7]
    ptotal[4,1] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]*C0[6]*(1-C0[7])
    ptotal[4,2] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]*(1-C0[6])*C0[7]
    ptotal[3,1] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[4,3] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])*C0[6]*C0[7]
    ptotal[3,2] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[3,3] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[2,2] += (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])*(1-C0[6])*(1-C0[7])

    ptotal[5,3] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]*C0[6]*C0[7]
    ptotal[4,2] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]*C0[6]*(1-C0[7])
    ptotal[4,3] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]*(1-C0[6])*C0[7]
    ptotal[3,2] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[4,4] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])*C0[6]*C0[7]
    ptotal[3,3] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[3,4] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[2,3] += (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])*(1-C0[6])*(1-C0[7])
    ptotal[4,2] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]*C0[6]*C0[7]
    ptotal[3,1] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]*C0[6]*(1-C0[7])
    ptotal[3,2] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]*(1-C0[6])*C0[7]
    ptotal[2,1] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[3,3] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])*C0[6]*C0[7]
    ptotal[2,2] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[2,3] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[1,2] += (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])*(1-C0[6])*(1-C0[7])

    ptotal[5,4] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]*C0[6]*C0[7]
    ptotal[4,3] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]*C0[6]*(1-C0[7])
    ptotal[4,4] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]*(1-C0[6])*C0[7]
    ptotal[3,3] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[4,5] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])*C0[6]*C0[7]
    ptotal[3,4] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[3,5] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[2,4] += (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])*(1-C0[6])*(1-C0[7])
    ptotal[4,3] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]*C0[6]*C0[7]
    ptotal[3,2] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]*C0[6]*(1-C0[7])
    ptotal[3,3] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]*(1-C0[6])*C0[7]
    ptotal[2,2] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[3,4] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])*C0[6]*C0[7]
    ptotal[2,3] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[2,4] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[1,3] += (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])*(1-C0[6])*(1-C0[7])
    ptotal[4,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]*C0[6]*C0[7]
    ptotal[3,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]*C0[6]*(1-C0[7])
    ptotal[3,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]*(1-C0[6])*C0[7]
    ptotal[2,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[3,5] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])*C0[6]*C0[7]
    ptotal[2,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[2,5] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])*(1-C0[6])*C0[7]
    ptotal[1,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])*(1-C0[6])*(1-C0[7])
    ptotal[3,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]*C0[6]*C0[7]
    ptotal[2,2] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]*C0[6]*(1-C0[7])
    ptotal[2,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]*(1-C0[6])*C0[7]
    ptotal[1,2] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]*(1-C0[6])*(1-C0[7])
    ptotal[2,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])*C0[6]*C0[7]
    ptotal[1,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])*C0[6]*(1-C0[7])
    ptotal[1,4] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])*(1-C0[6])*C0[7]
    # ptotal[0,3] += (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])*(1-C0[6])*(1-C0[7])

    ptotal0 = (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])*(1-C0[6])*(1-C0[7])
    test0 = zeros(Float64,1,5)
    test0[1,3] = ptotal0
    ptotal = [test0;ptotal]
    # ptotal = ptotal .+ 1e-15

    Sobs = 0.0
    for jj1 = 1:6
        for jj2 = 1:5
            pop = ptotal[jj1,jj2]
            if abs(pop) != 0.0
               Sobs += -pop*log(pop)
            end
        end
    end

    # return ptotal
    return Sobs

end

##################### functions not needed anymore

function energy2index(Ej::Float64,arrayE0::Vector{Float64},Nene::Int64)

    diff = 0.0

    for jj = 1:Nene
        diff = abs(Ej - arrayE0[jj])
        if diff < 10^(-10)
           return jj
        end
    end

end

function calculatep_test0(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # this function works only for thermal states

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # the following calculation for the probability replies on the fact that epsilonL and epsilonR are equally spacing

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    epsilon = diag(matH)
    # epsilonL = epsilon[2:K+1]
    # epsilonR = epsilon[K+2:2*K+1]
    ind_epsilon = zeros(Int64,2*K+1)
    epsilon_sort = zeros(Float64,2*K+1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    diag_Ct = zeros(Float64,2*K+1,Nt)

    Nene = 2^10 #Int64(K^2/2)+1
    # Nene = Nenebath*2+1 #(Int64(K/2)^2+1)*2+1
    ptotal = zeros(Float64,2*K+1,Nene,Nt)
    ptotal_part = zeros(Float64,2*K+1)
    ptotal_part_comb = zeros(Float64,2*K+1)
    criterion = 0.1

    count_total1 = 0
    count_total0 = 0
    ind = 0

    counteps_total1 = zeros(Int64,2*K+1)
    counteps_total0 = zeros(Int64,2*K+1)
    counteps_total = zeros(Int64,2*K+1)

    arrayE = zeros(Float64,Nene)
    indE = 0

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # Tr[n_j rho] for j = L,R
        diag_Ct[:,tt] .= real(diag(Ct[1:2*K+1,1:2*K+1]))
        # diag_Ct_L[:,tt] .= real(diag(Ct[2:K+1,2:K+1]))
        # diag_Ct_R[:,tt] .= real(diag(Ct[K+2:end,K+2:end]))

        ind_epsilon .= sortperm(epsilon)
        epsilon_sort .= epsilon[ind_epsilon]
        diag_Ct[:,tt] .= diag_Ct[ind_epsilon,tt]

        # count the number of Tr[n_j rho] close to 1 or 0
        ptotal_part .= 0.0
        count_total1 = 0
        counteps_total1 .= 0
        count_total0 = 0
        counteps_total0 .= 0
        ind = 0
        counteps_total .= 0
        for jj = 1:2*K+1
            if diag_Ct[jj,tt] > 1.0 - criterion
               ptotal_part[jj] = 1.0
               count_total1 += 1
               counteps_total1[count_total1] = jj
            elseif diag_Ct[jj,tt] < criterion
               ptotal_part[jj] = 1.0 - 0.0
               count_total0 += 1
               counteps_total0[count_total0] = jj
            else
               ptotal_part[jj] = 1.0 - diag_Ct[jj,tt]
               ind += 1
               counteps_total[ind] = jj
            end
        end

        # for N_j = count_total1 and E_j = sum(epsilon_sort[counteps_total1[1:count_total1]])
        indE = 1
        ptotal[count_total1,indE,tt] = prod(ptotal_part[:])
        arrayE[indE] = sum(epsilon_sort[counteps_total1[1:count_total1]])

        for jjN = 1:2*K+1-count_total0-count_total1 #count_total1+1:2*K+1-count_total0

            combind = collect(combinations(counteps_total[1:ind],jjN))
            Mcombind = length(combind)
            for iiN = 1:Mcombind
                ptotal_part_comb .= ptotal_part
                for kkN = 1:jjN
                    ptotal_part_comb[combind[iiN][kkN]] = diag_Ct[combind[iiN][kkN],tt]
                end
                indE += 1
                ptotal[count_total1+jjN,indE,tt] += prod(ptotal_part_comb[:])
                arrayE[indE] = E0 + sum(epsilon_sort[combind[iiN]])
            end

        end



    end

    Sobs = zeros(Float64,Nt)
    for tt = 1:Nt
        for jjN = count_total1+1:2*K+1-count_total0
            for jjE = 1:Nene
                pop = ptotal[jjN,jjE,tt]
                if abs(pop) != 0.0
                   Sobs[tt] += -pop*log(pop)
                end
            end
        end
    end


    return time, arrayE0, ptotal, Sobs

    # technically, N_j=0 and E_j=0 should be considered, but let me ignore it since p_{0,0}=0

end

function calculatep_test_parfor(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # epsilonLR = zeros(ComplexF64,K*2+1)
    epsilonLR = diag(matH)
    epsilonL = epsilonLR[2:K+1]
    epsilonR = epsilonLR[K+2:2*K+1]
    epsilonL_tilde = zeros(Float64,K,Nt)

    # Ct = zeros(ComplexF64,K*2+1,K*2+1)
    # diag_Ct_L = zeros(Float64,K,Nt)
    # diag_Ct_R = zeros(Float64,K,Nt)
    # eigval_Ct_L = zeros(Float64,K)
    # eigvec_Ct_L = zeros(Float64,K,K)

    # Nenebath = Int64(K*(K+1)/2)
    lengthErange = 2^23
    # pL = zeros(Float64,K,lengthErange)
    pLround = zeros(Float64,K,lengthErange,Nt)
    # pL_part = zeros(Float64,K)
    # pL_part_comb = zeros(Float64,K)
    criterion = 0.1

    # count_L = zeros(Int64,K)
    # count_L1 = 0
    # count_L0 = 0

    # ind = 0
    # counteps_L1 = zeros(Int64,K)
    # counteps_L0 = 0
    # indE = 0
    # arrayE = zeros(Float64,lengthErange)
    arrayEround = zeros(Float64,lengthErange,Nt)
    arrayEroundsize = zeros(Int64,Nt)
    arrayEsize = zeros(Int64,Nt)
    arrayN = zeros(Int64,2,Nt)

    Depsilon = W/(K-1)
    epsilonLposi = Array(1:K)*Depsilon #(kk-1)*Depsilon - W/2

    Threads.@threads for tt = 1:Nt
    # for tt = 1:Nt

        println("t=",tt)

        # @time begin

        # time evolution of correlation matrix
        Ctparfor = vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ctparfor .= Ctparfor*C0
        Ctparfor .= Ctparfor*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # Tr[tilde{n}_j rho] for j = L,R
        lambda, eigvec_Ct_L = eigen(Ctparfor[2:K+1,2:K+1],sortby = x -> -abs(x))
        eigval_Ct_L = real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:K
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilonLposi)
        end
        indss = sortperm(epsilonL_tilde[:,tt])
        epsilonL_tilde[:,tt] .= epsilonL_tilde[indss,tt]
        eigval_Ct_L .= eigval_Ct_L[indss]

        # count the number of Tr[n_j rho] close to 1 or 0
        pL_part = zeros(Float64,K) # pL_part .= 0.0
        count_L = zeros(Int64,K) #count_L .= 0
        count_L1 = 0
        count_L0 = 0
        ind = 0
        # ind1 = 0
        # ind0 = 0
        counteps_L1 = zeros(Int64,K) #counteps_L1 .= 0
        # counteps_L0 .= 0
        for jj = 1:K
            if eigval_Ct_L[jj] > 1.0 - criterion
               pL_part[jj] = 1.0
               count_L1 += 1
               counteps_L1[count_L1] = jj
            elseif eigval_Ct_L[jj] < criterion
               pL_part[jj] = 1.0 - 0.0
               count_L0 += 1
               # ind0 += 1
               # counteps_L0[ind0] = jj
            else
               pL_part[jj] = 1.0 - eigval_Ct_L[jj]
               ind += 1
               count_L[ind] = jj
            end
        end

        Esize = 2^ind

        # the probability is zero for N_j < count_L1 or N_j > count_L0
        # pL[1:count_L1-1,:,tt] .= 0.0
        # pL[K-count_L0+1:end,:,tt] .= 0.0

        # the probability is zero for E_j < counteps_L1 or E_j > Nenebath-counteps_L0
        # pL[:,Esize+1:end,tt] .= 0.0

        pL = spzeros(Float64,K,lengthErange) #zeros(Float64,K,lengthErange)
        arrayE = spzeros(Float64,lengthErange) #zeros(Float64,lengthErange)
        indE = 1
        pL_part_comb = pL_part
        pL[count_L1,indE] = prod(pL_part[:]) # for N_j = count_L1 and E_j = counteps_L1
        arrayE0 = sum(epsilonL_tilde[counteps_L1[1:count_L1],tt])
        arrayE[indE] = 0.0+arrayE0
        arrayN[1,tt] = count_L1
        arrayN[2,tt] = count_L1+ind

        for jjN = 1:ind

            combind = collect(combinations(count_L[1:ind],jjN))
            Mcombind = length(combind)

            for iiN = 1:Mcombind
                pL_part_comb .= pL_part
                for kkN = 1:jjN
                    pL_part_comb[combind[iiN][kkN]] = eigval_Ct_L[combind[iiN][kkN]]
                end
                indE += 1
                pL[count_L1+jjN,indE] = prod(pL_part_comb[:])
                arrayE[indE] = sum(epsilonL_tilde[combind[iiN],tt])+arrayE0
            end

        end

        indarrayE = sortperm(arrayE[1:indE])
        arrayE[1:indE] = arrayE[indarrayE]
        for jjN = 1:ind
            pL[count_L1+jjN,1:indE] = pL[count_L1+jjN,indarrayE]
        end
        arrayEsize[tt] = indE

        # round E
        check0 = arrayE[1]
        indcheck0 = 0
        indround = 0
        for jjE = 2:indE

            if arrayE[jjE]-check0 < 10^(-10)*check0
               indcheck0 += 1
            else
               indround += 1
               arrayEround[indround,tt] = check0
               for jjN = 1:ind
                   pLround[count_L1+jjN,indround,tt] = sum(pL[count_L1+jjN,jjE-1-indcheck0:jjE-1])
               end
               check0 = arrayE[jjE]
               indcheck0 = 0
            end

        end
        indround += 1
        arrayEround[indround,tt] = check0 #arrayE[indE]
        for jjN = 1:ind
            pLround[count_L1+jjN,indround,tt] = sum(pL[count_L1+jjN,indE-indcheck0:indE])
        end
        arrayEroundsize[tt] = indround

        # end

    end

    # return time, arrayE, arrayEsize, pL, arrayEround, arrayEroundsize, pLround
    return time, arrayEround, arrayEroundsize, arrayN, pLround

    # technically, N_j=0 and E_j=0 should be considered, but let me ignore it since p_{0,0}=0

end

function calculatepL_test(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # epsilonLR = zeros(ComplexF64,K*2+1)
    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilonL_tilde = zeros(Float64,K,Nt)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    # diag_Ct_L = zeros(Float64,K,Nt)
    # diag_Ct_R = zeros(Float64,K,Nt)
    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    # eigval_Ct = zeros(Float64,K)
    # eigvec_Ct = zeros(Float64,K,K)

    # Nenebath = Int64(K*(K+1)/2)
    lengthErange = 2^21 #2^23
    pL = zeros(Float64,K,lengthErange) #spzeros(Float64,K,lengthErange)
    pLround = zeros(Float64,K,lengthErange,Nt)
    pL_part = zeros(Float64,K)
    pL_part_comb = zeros(Float64,K)
    criterion = 0.1

    count_L = zeros(Int64,K)
    count_L1 = 0
    count_L0 = 0

    ind = 0
    counteps_L1 = zeros(Int64,K)
    # counteps_L0 = 0
    indE = 0
    arrayE = zeros(Float64,lengthErange) #spzeros(Float64,lengthErange)
    arrayEround = zeros(Float64,lengthErange,Nt)
    arrayEroundsize = zeros(Int64,Nt)
    arrayEsize = zeros(Int64,Nt)
    arrayN = zeros(Int64,2,Nt)

    Depsilon = W/(K-1)
    epsilonLposi = Array(1:K)*Depsilon #(kk-1)*Depsilon - W/2
    # epsilonRposi = Array(1:K)*Depsilon #(kk-1)*Depsilon - W/2
    # epsilonposi = [0.0; epsilonLposi; epsilonRposi]

    for tt = 1:Nt

        println("t=",tt)

        @time begin

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # Tr[n_j rho] for j = L,R
        # diag_Ct_L[:,tt] .= real(diag(Ct[2:K+1,2:K+1]))
        # diag_Ct_R[:,tt] .= real(diag(Ct[K+2:end,K+2:end]))

        # Tr[tilde{n}_j rho] for j = L,R
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1],sortby = x -> -abs(x))
        eigval_Ct_L .= real.(lambda)

        # Tr[tilde{n}_total rho]
        # lambda, eigvec_Ct = eigen(Ct[1:2*K+1,1:2*K+1],sortby = x -> -abs(x))
        # eigval_Ct .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:K
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilonL)
        end
        # indss = sortperm(epsilonL_tilde[:,tt])
        # epsilonL_tilde[:,tt] .= epsilonL_tilde[indss,tt]
        # eigval_Ct_L .= eigval_Ct_L[indss]
        # for ss = 1:2*K+1
            # epsilon_tilde[ss,tt] = sum(abs.(eigvec_Ct[:,ss]).^2 .* epsilonposi)
        # end

        # count the number of Tr[n_j rho] close to 1 or 0
        pL_part .= 0.0
        count_L .= 0
        count_L1 = 0
        count_L0 = 0
        ind = 0
        # ind1 = 0
        # ind0 = 0
        counteps_L1 .= 0
        # counteps_L0 .= 0
        for jj = 1:K
            if eigval_Ct_L[jj] > 1.0 - criterion
               pL_part[jj] = 1.0
               count_L1 += 1
               counteps_L1[count_L1] = jj
            elseif eigval_Ct_L[jj] < criterion
               pL_part[jj] = 1.0 - 0.0
               count_L0 += 1
               # ind0 += 1
               # counteps_L0[ind0] = jj
            else
               pL_part[jj] = 1.0 - eigval_Ct_L[jj]
               ind += 1
               count_L[ind] = jj
            end
        end

        Esize = 2^ind

        # the probability is zero for N_j < count_L1 or N_j > count_L0
        # pL[1:count_L1-1,:,tt] .= 0.0
        # pL[K-count_L0+1:end,:,tt] .= 0.0

        # the probability is zero for E_j < counteps_L1 or E_j > Nenebath-counteps_L0
        # pL[:,Esize+1:end,tt] .= 0.0

        pL .= 0.0
        arrayE .= 0.0
        indE = 1
        pL[count_L1,indE] = prod(pL_part[:]) # for N_j = count_L1 and E_j = counteps_L1
        arrayE0 = sum(epsilonL_tilde[counteps_L1[1:count_L1],tt])
        arrayE[indE] = 0.0+arrayE0
        arrayN[1,tt] = count_L1
        arrayN[2,tt] = count_L1+ind

        for jjN = 1:ind

            combind = collect(combinations(count_L[1:ind],jjN))
            Mcombind = length(combind)

            for iiN = 1:Mcombind
                pL_part_comb .= pL_part
                for kkN = 1:jjN
                    pL_part_comb[combind[iiN][kkN]] = eigval_Ct_L[combind[iiN][kkN]]
                end
                indE += 1
                pL[count_L1+jjN,indE] = prod(pL_part_comb[:])
                arrayE[indE] = sum(epsilonL_tilde[combind[iiN],tt])+arrayE0
            end

        end

        indarrayE = sortperm(arrayE[1:indE])
        arrayE[1:indE] = arrayE[indarrayE]
        for jjN = 1:ind
            pL[count_L1+jjN,1:indE] = pL[count_L1+jjN,indarrayE]
        end
        arrayEsize[tt] = indE

        # round E
        check0 = arrayE[1]
        indcheck0 = 0
        indround = 0
        for jjE = 2:indE

            if arrayE[jjE]-check0 < 10^(-10)*check0
               indcheck0 += 1
            else
               indround += 1
               arrayEround[indround,tt] = check0
               for jjN = 1:ind
                   pLround[count_L1+jjN,indround,tt] = sum(pL[count_L1+jjN,jjE-1-indcheck0:jjE-1])
               end
               check0 = arrayE[jjE]
               indcheck0 = 0
            end

        end
        indround += 1
        arrayEround[indround,tt] = check0 #arrayE[indE,tt]
        for jjN = 1:ind
            pLround[count_L1+jjN,indround,tt] = sum(pL[count_L1+jjN,indE-indcheck0:indE])
        end
        arrayEroundsize[tt] = indround

        end

    end

    # return time, arrayE, arrayEsize, pL, arrayEround, arrayEroundsize, pLround
    return time, arrayEround, arrayEroundsize, arrayN, pLround

    # technically, N_j=0 and E_j=0 should be considered, but let me ignore it since p_{0,0}=0

end

function calculateptotal_test(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilon_tilde = zeros(Float64,2*K+1,Nt)
    indtilde = zeros(Int64,2*K+1)

    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)

    # Nenebath = Int64(K*(K+1)/2)
    lengthErange = 2^22 #2^21
    ptotal = zeros(Float64,2*K+1+1,lengthErange)
    ptotalround = zeros(Float64,2*K+1+1,lengthErange,Nt)
    ptotal_part = zeros(Float64,2*K+1)
    ptotal_part_comb = zeros(Float64,2*K+1)
    criterion = 0.00001

    count_total = zeros(Int64,2*K+1)
    count_total1 = 0
    count_total0 = 0
    counteps_total1 = zeros(Int64,2*K+1)
    ind = 0

    indE = 0
    arrayE = zeros(Float64,lengthErange) #spzeros(Float64,lengthErange)
    arrayEround = zeros(Float64,lengthErange,Nt)
    arrayEroundsize = zeros(Int64,Nt)
    arrayEsize = zeros(Int64,Nt)
    arrayN = zeros(Int64,2,Nt)

    for tt = 1:Nt

        println("t=",tt)

        @time begin

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        lambda, eigvec_Ct = eigen(Ct) #eigen(Ct,sortby = x -> -abs(x))
        eigval_Ct .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:2*K+1
            epsilon_tilde[ss,tt] = sum(abs.(eigvec_Ct[:,ss]).^2 .* epsilon)
        end
        indtilde .= sortperm(epsilon_tilde[:,tt])
        epsilon_tilde[:,tt] .= epsilon_tilde[indtilde,tt]
        eigval_Ct .= eigval_Ct[indtilde]

        # eigval_Ct .= diag(C0)
        # epsilon_tilde[:,tt] = epsilon
        # indtilde .= sortperm(epsilon_tilde[:,tt])
        # epsilon_tilde[:,tt] .= epsilon_tilde[indtilde,tt]
        # eigval_Ct .= eigval_Ct[indtilde]

        # count the number of Tr[n_j rho] close to 1 or 0
        ptotal_part .= 0.0
        count_total .= 0
        count_total1 = 0
        count_total0 = 0
        ind = 0
        counteps_total1 .= 0

        for jj = 1:2*K+1
            if eigval_Ct[jj] > 1.0 - criterion
               ptotal_part[jj] = 1.0
               count_total1 += 1
               counteps_total1[count_total1] = jj
            elseif eigval_Ct[jj] < criterion
               ptotal_part[jj] = 1.0 - 0.0
               count_total0 += 1
            else
               ptotal_part[jj] = 1.0 - eigval_Ct[jj]
               ind += 1
               count_total[ind] = jj
            end
        end

        ptotal .= 0.0
        arrayE .= 0.0
        indE = 0
        arrayE0 = 0.0
        indE += 1
        ptotal[1+count_total1,indE] = prod(ptotal_part[:])
        arrayE0 = sum(epsilon_tilde[counteps_total1[1:count_total1],tt])
        arrayE[indE] = arrayE0
        arrayN[1,tt] = count_total1
        arrayN[2,tt] = count_total1+ind

        for jjN = 1:ind

            combind = collect(combinations(count_total[1:ind],jjN))
            Mcombind = length(combind)

            for iiN = 1:Mcombind
                ptotal_part_comb .= ptotal_part
                for kkN = 1:jjN
                    ptotal_part_comb[combind[iiN][kkN]] = eigval_Ct[combind[iiN][kkN]]
                end
                indE += 1
                ptotal[1+count_total1+jjN,indE] += prod(ptotal_part_comb[:])
                arrayE[indE] = sum(epsilon_tilde[combind[iiN],tt])+arrayE0
            end

        end

        indarrayE = sortperm(arrayE[1:indE])
        arrayE[1:indE] = arrayE[indarrayE]
        for jjN = 0:ind
            ptotal[1+count_total1+jjN,1:indE] = ptotal[1+count_total1+jjN,indarrayE]
        end
        arrayEsize[tt] = indE

        # round E
        check0 = arrayE[1]
        indcheck0 = 0
        indround = 0
        for jjE = 2:indE

            if abs(arrayE[jjE]-check0) < 10^(-10)
               indcheck0 += 1
            else
               indround += 1
               arrayEround[indround,tt] = check0

               for jjN = 0:ind
                   ptotalround[1+count_total1+jjN,indround,tt] = sum(ptotal[1+count_total1+jjN,jjE-1-indcheck0:jjE-1])
               end
               check0 = arrayE[jjE]
               indcheck0 = 0
            end

        end
        indround += 1
        arrayEround[indround,tt] = check0
        for jjN = 0:ind
            ptotalround[1+count_total1+jjN,indround,tt] = sum(ptotal[1+count_total1+jjN,indE-indcheck0:indE])
        end
        arrayEroundsize[tt] = indround

        end

    end

    return time, arrayEround, arrayEroundsize, arrayN, ptotalround

end

function calculatepLR_test2(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=2

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilonL_tilde = zeros(Float64,K,Nt)
    epsilonR_tilde = zeros(Float64,K,Nt)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)

    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    arrayELR = [-W/2,0.0,W/2]
    pL = zeros(Float64,2,3)
    pR = zeros(Float64,2,3)

    println(arrayELR)
    # println(C0)

    pL[2,2] += C0[2]*C0[3]
    pL[1,1] += C0[2]*(1-C0[3])

    pL[1,3] += (1-C0[2])*C0[3]
    # pL[0,2] += (1-C0[2])*(1-C0[3])
    pL0 = (1-C0[2])*(1-C0[3])

    pR[2,2] += C0[4]*C0[5]
    pR[1,1] += C0[4]*(1-C0[5])

    pR[1,3] += (1-C0[4])*C0[5]
    # pR[0,2] += (1-C0[4])*(1-C0[5])
    pR0 = (1-C0[4])*(1-C0[5])

    test0 = zeros(Float64,1,3)
    test0[1,2] = pL0
    pL = [test0;pL]

    test0 = zeros(Float64,1,3)
    test0[1,2] = pR0
    pR = [test0;pR]

    SobsL = 0.0
    SobsR = 0.0
    for jj1 = 1:3
        for jj2 = 1:3
            pop = pL[jj1,jj2]
            if abs(pop) != 0.0
               SobsL += -pop*log(pop)
            end
            pop = pR[jj1,jj2]
            if abs(pop) != 0.0
               SobsR += -pop*log(pop)
            end
        end
    end
    println(SobsL)
    println(SobsR)

    C0 = diagm(C0)

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:K
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilonL)
            epsilonR_tilde[ss,tt] = sum(abs.(eigvec_Ct_R[:,ss]).^2 .* epsilonR)
        end

        indL = sortperm(epsilonL_tilde[:,tt])
        epsilonL_tilde[:,tt] = epsilonL_tilde[indL,tt]
        eigval_Ct_L = eigval_Ct_L[indL]
        arrayEL = [epsilonL_tilde[1,tt],sum(epsilonL_tilde[:,tt]),epsilonL_tilde[2,tt]]
        println(arrayEL)

        indR = sortperm(epsilonR_tilde[:,tt])
        epsilonR_tilde[:,tt] = epsilonR_tilde[indR,tt]
        eigval_Ct_R = eigval_Ct_R[indR]
        arrayER = [epsilonR_tilde[1,tt],sum(epsilonR_tilde[:,tt]),epsilonR_tilde[2,tt]]
        println(arrayER)

        pL .= 0.0
        pL[2,2] += eigval_Ct_L[1]*eigval_Ct_L[2]
        pL[1,1] += eigval_Ct_L[1]*(1-eigval_Ct_L[2])
        pL[1,3] += (1-eigval_Ct_L[1])*eigval_Ct_L[2]
        pL0 = (1-eigval_Ct_L[1])*(1-eigval_Ct_L[2])
        test0 = zeros(Float64,1,3)
        test0[1,2] = pL0
        pL = [test0;pL]

        pR .= 0.0
        pR[2,2] += eigval_Ct_R[1]*eigval_Ct_R[2]
        pR[1,1] += eigval_Ct_R[1]*(1-eigval_Ct_R[2])
        pR[1,3] += (1-eigval_Ct_R[1])*eigval_Ct_R[2]
        pR0 = (1-eigval_Ct_R[1])*(1-eigval_Ct_R[2])
        test0 = zeros(Float64,1,3)
        test0[1,2] = pR0
        pR = [test0;pR]

        SobsL = 0.0
        SobsR = 0.0
        for jj1 = 1:3
            for jj2 = 1:3
                pop = pL[jj1,jj2]
                if abs(pop) != 0.0
                   SobsL += -pop*log(pop)
                end
                pop = pR[jj1,jj2]
                if abs(pop) != 0.0
                   SobsR += -pop*log(pop)
                end
            end
        end
        println(SobsL)
        println(SobsR)

    end

    # return SobsL, SobsR

end

function calculatepLR_test4(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # only for K=3

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = LinRange(0.0,tf,Nt)

    # correlation matrix
    # at initial
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 = NaN
    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end

    epsilon = diag(matH)
    epsilonL = epsilon[2:K+1]
    epsilonR = epsilon[K+2:2*K+1]
    epsilonL_tilde = zeros(Float64,K,Nt)
    epsilonR_tilde = zeros(Float64,K,Nt)

    Ct = zeros(ComplexF64,2*K+1,2*K+1)

    eigval_Ct_L = zeros(Float64,K)
    eigvec_Ct_L = zeros(Float64,K,K)
    eigval_Ct_R = zeros(Float64,K)
    eigvec_Ct_R = zeros(Float64,K,K)

    arrayELR = [-W/2,0.0,W/2]
    pL = zeros(Float64,K,3)
    pR = zeros(Float64,K,3)

    println(arrayELR)
    # println(C0)

    pL[3,2] += C0[2]*C0[3]*C0[4]
    pL[2,1] += C0[2]*C0[3]*(1-C0[4])
    pL[2,2] += C0[2]*(1-C0[3])*C0[4]
    pL[1,1] += C0[2]*(1-C0[3])*(1-C0[4])
    pL[2,3] += (1-C0[2])*C0[3]*C0[4]
    pL[1,2] += (1-C0[2])*C0[3]*(1-C0[4])
    pL[1,3] += (1-C0[2])*(1-C0[3])*C0[4]
    # pL[0,0] += (1-C0[2])*(1-C0[3])*(1-C0[4])
    pL0 = (1-C0[2])*(1-C0[3])*(1-C0[4])

    test0 = zeros(Float64,1,3)
    test0[1,2] = pL0
    pL = [test0;pL]

    pR[3,2] += C0[5]*C0[6]*C0[7]
    pR[2,1] += C0[5]*C0[6]*(1-C0[7])
    pR[2,2] += C0[5]*(1-C0[6])*C0[7]
    pR[1,1] += C0[5]*(1-C0[6])*(1-C0[7])
    pR[2,3] += (1-C0[5])*C0[6]*C0[7]
    pR[1,2] += (1-C0[5])*C0[6]*(1-C0[7])
    pR[1,3] += (1-C0[5])*(1-C0[6])*C0[7]
    # pR[0,0] += (1-C0[5])*(1-C0[6])*(1-C0[7])
    pR0 = (1-C0[5])*(1-C0[6])*(1-C0[7])

    test0 = zeros(Float64,1,3)
    test0[1,2] = pR0
    pR = [test0;pR]

    SobsL = 0.0
    SobsR = 0.0
    for jj1 = 1:4
        for jj2 = 1:3
            pop = pL[jj1,jj2]
            if abs(pop) != 0.0
               SobsL += -pop*log(pop)
            end
            pop = pR[jj1,jj2]
            if abs(pop) != 0.0
               SobsR += -pop*log(pop)
            end
        end
    end
    println(SobsL)
    println(SobsR)

    C0 = diagm(C0)

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # L
        lambda, eigvec_Ct_L = eigen(Ct[2:K+1,2:K+1])
        eigval_Ct_L .= real.(lambda)

        # R
        lambda, eigvec_Ct_R = eigen(Ct[K+2:2*K+1,K+2:2*K+1])
        eigval_Ct_R .= real.(lambda)

        # tiltde{epsilon}, epsilon in the a basis
        for ss = 1:K
            epsilonL_tilde[ss,tt] = sum(abs.(eigvec_Ct_L[:,ss]).^2 .* epsilonL)
            epsilonR_tilde[ss,tt] = sum(abs.(eigvec_Ct_R[:,ss]).^2 .* epsilonR)
        end

        indL = sortperm(epsilonL_tilde[:,tt])
        epsilonL_tilde[:,tt] = epsilonL_tilde[indL,tt]
        eigval_Ct_L = eigval_Ct_L[indL]
        arrayEL = [epsilonL_tilde[1,tt],sum(epsilonL_tilde[:,tt]),epsilonL_tilde[2,tt]]
        println(arrayEL)

        indR = sortperm(epsilonR_tilde[:,tt])
        epsilonR_tilde[:,tt] = epsilonR_tilde[indR,tt]
        eigval_Ct_R = eigval_Ct_R[indR]
        arrayER = [epsilonR_tilde[1,tt],sum(epsilonR_tilde[:,tt]),epsilonR_tilde[2,tt]]
        println(arrayER)

        pL .= 0.0
        pL[3,2] += eigval_Ct_L[1]*eigval_Ct_L[2]*eigval_Ct_L[3]
        pL[2,1] += eigval_Ct_L[1]*eigval_Ct_L[2]*(1-eigval_Ct_L[3])
        pL[2,2] += eigval_Ct_L[1]*(1-eigval_Ct_L[2])*eigval_Ct_L[3]
        pL[1,1] += eigval_Ct_L[1]*(1-eigval_Ct_L[2])*(1-eigval_Ct_L[3])
        pL[2,3] += (1-eigval_Ct_L[1])*eigval_Ct_L[2]*eigval_Ct_L[3]
        pL[1,2] += (1-eigval_Ct_L[1])*eigval_Ct_L[2]*(1-eigval_Ct_L[3])
        pL[1,3] += (1-eigval_Ct_L[1])*(1-eigval_Ct_L[2])*eigval_Ct_L[3]
        pL0 = (1-eigval_Ct_L[1])*(1-eigval_Ct_L[2])*(1-eigval_Ct_L[3])
        test0 = zeros(Float64,1,3)
        test0[1,2] = pL0
        pL = [test0;pL]

        pR .= 0.0
        pR[3,2] += eigval_Ct_R[1]*eigval_Ct_R[2]*eigval_Ct_R[3]
        pR[2,1] += eigval_Ct_R[1]*eigval_Ct_R[2]*(1-eigval_Ct_R[3])
        pR[2,2] += eigval_Ct_R[1]*(1-eigval_Ct_R[2])*eigval_Ct_R[3]
        pR[1,1] += eigval_Ct_R[1]*(1-eigval_Ct_R[2])*(1-eigval_Ct_R[3])
        pR[2,3] += (1-eigval_Ct_R[1])*eigval_Ct_R[2]*eigval_Ct_R[3]
        pR[1,2] += (1-eigval_Ct_R[1])*eigval_Ct_R[2]*(1-eigval_Ct_R[3])
        pR[1,3] += (1-eigval_Ct_R[1])*(1-eigval_Ct_R[2])*eigval_Ct_R[3]
        pR0 = (1-eigval_Ct_R[1])*(1-eigval_Ct_R[2])*(1-eigval_Ct_R[3])
        test0 = zeros(Float64,1,3)
        test0[1,2] = pR0
        pR = [test0;pR]

        SobsL = 0.0
        SobsR = 0.0
        for jj1 = 1:4
            for jj2 = 1:3
                pop = pL[jj1,jj2]
                if abs(pop) != 0.0
                   SobsL += -pop*log(pop)
                end
                pop = pR[jj1,jj2]
                if abs(pop) != 0.0
                   SobsR += -pop*log(pop)
                end
            end
        end
        println(SobsL)
        println(SobsR)

    end

    # return SobsL, SobsR

end

function calculateSobs_test(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    time, arrayEround, arrayEroundsize, arrayN, pround = calculateptotal_test3(K,W,betaL,betaR,GammaL,GammaR,muL,muR,tf,Nt)
    # time, arrayEround, arrayEroundsize, arrayN, pround = calculatepL_test(K,W,betaL,betaR,GammaL,GammaR,muL,muR,tf,Nt)

    Sobs = zeros(Float64,Nt)

    for tt = 1:Nt
        for jjN = arrayN[1,tt]:arrayN[2,tt]
            for jjE = 1:arrayEroundsize[tt]
                pop = pround[1+jjN,jjE,tt]
                if abs(pop) != 0.0
                   Sobs[tt] += -pop*log(pop)
                end
            end
        end
    end

    return time, arrayEround, arrayEroundsize, arrayN, pround, Sobs

end
