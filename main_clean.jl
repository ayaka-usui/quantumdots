
using Arpack, SparseArrays, LinearAlgebra
using NLsolve
using Plots
using Distributions, Random
using JLD
using Combinatorics
using LaTeXStrings

# sub functions used in "dynamics" ###################
function createH_differentKLKR!(epsilond::Float64,KL::Int64,KR::Int64,W::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # construct Hamiltonian

    # matH = sparse(Float64,K*2+1,K*2+1)
    matH .= 0.0
    DepsilonL = W/(KL-1)
    DepsilonR = W/(KR-1)
    tunnelL = sqrt(GammaL*DepsilonL/(2*pi))
    tunnelR = sqrt(GammaR*DepsilonR/(2*pi))

    matH[1,1] = epsilond # epsilon for the system

    for kk = 1:KL
        matH[1+kk,1+kk] = (kk-1)*DepsilonL - W/2 # epsilon for the bath L
        matH[1+kk,1] = tunnelL # tunnel with the bath L
    end
    for kk = 1:KR
        matH[1+KL+kk,1+KL+kk] = (kk-1)*DepsilonR - W/2 # epsilon for the bath R
        matH[1+KL+kk,1] = tunnelR # tunnel with the bath R
    end

    matH .= matH + matH' - spdiagm(diag(matH)) # Hermitian

end

function distribute_timepoint(Nt::Int64,ti::Float64,tf::Float64)

    # distribute points at regular intervals in log plot

    if ti == 0.0 
       ti = 1e-4 # remove 0 to avoid log 0
    end

    ti_log10 = log10(ti)
    tf_log10 = log10(tf)

    time_log10 = LinRange(ti_log10,tf_log10,Nt)
    time = 10.0.^(time_log10)

    return time

end

function set_pureinitialstate(KL::Int64,KR::Int64,nF::Vector{Float64})

    # set pure random initial state
    # see the article for the details of how it is set

    seed = 5 # can be any integer, but "5" produces the same results as the article
    Random.seed!(seed)
    npure = rand(Uniform(0,1), KL+KR)

    for kk = 1:KL+KR
        if npure[kk] <= nF[kk]
           npure[kk] = 1
        else #npure[kk] > nF[kk]
           npure[kk] = 0
        end
    end

    return npure

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

function funeffectivebetamu(epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64)

    # estimate inverse temperature and chemical potential of bath state L or R
    sol = nlsolve((F,x) ->funbetamu!(F,x,epsilon,Ene,Np), [beta0; mu0])
    return sol.zero

end

function funeffectivebetamu2(epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64,Cgg::Vector{Float64},matCgg::Matrix{Float64},val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64})

    # estimate inverse temperature and chemical potential of global Gibbs state
    sol = nlsolve((F,x) ->funbetamu2!(F,x,epsilon,Ene,Np,Cgg,matCgg,val_matH,vec_matH,invvec_matH), [beta0; mu0])
    return sol.zero

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

function vNEfrommatC(val_matC::Union{Vector{Float64},Float64})

    # compute vNE from the eigenvalues val_matC of correlation matrix

    vNE = 0.0

    for jj = 1:length(val_matC)
        if val_matC[jj] > 0 && val_matC[jj] < 1 # to avoid NaN from log (-1)
           vNE += -val_matC[jj]*log(val_matC[jj]) - (1.0-val_matC[jj])*log(1.0-val_matC[jj])
        end
    end

    return vNE

end

function compute_vNEpi(epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # compute vNE of the thermal state at beta and mu

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

    return vNEpi

end

# main code ##################
function dynamics(epsilond::Float64,KL::Int64,KR::Int64,W::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,ti::Float64,tf::Float64,Nt::Int64,pure::Int64)

    # Hamiltonian
    matH = spzeros(Float64,KL+KR+1,KL+KR+1)
    createH_differentKLKR!(epsilond,KL,KR,W,betaL,betaR,GammaL,GammaR,matH)
    epsilonLR = diag(Array(matH)) # energy level
    tLRk = matH[1,1:end] # tunnel coupling strength
    matH = Hermitian(Array(matH)) # Hamiltonian is hermitian
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # time
    time = distribute_timepoint(Nt-1,ti,tf) # distribute points at regular intervals in log plot
    pushfirst!(time,0.0) # time[1] = 0.0

    # initial correlation matrix
    # thermal state
    C0 = zeros(Float64,KL+KR+1)
    C0[1] = 0.0 + 1e-15 # n_d(0) # make it not 0 exactly to avoid 0.0 log 0.0 -> NaN
    for kk = 1:KL
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
    end
    for kk = 1:KR
        C0[1+KL+kk] = 1.0/(exp((matH[1+KL+kk,1+KL+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)

    # pure state
    if pure == 1
       nF = diag(C0[2:end,2:end])
       npure = set_pureinitialstate(KL,KR,nF)
       C0[2:end,2:end] = diagm(npure)
    end
    
    # estimate inverse temperature and chemical potential of global Gibbs state (see also Appendix B in the article)
    dC0 = diag(C0)
    E_tot0 = sum(dC0[1:KL+KR+1].*epsilonLR[1:KL+KR+1]) # total energy
    N_tot0 = sum(dC0[1:KL+KR+1]) # total particle number
    Cgg0 = zeros(Float64,KL+KR+1)
    matCgg0 = zeros(Float64,KL+KR+1,KL+KR+1)
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,val_matH,vec_matH,invvec_matH) # temperature and chemical potential of global Gibbs state
    println("beta_gg=",effpara0[1])
    println("mu_gg=",effpara0[2])

    # define global Gibbs state
    Cgg = globalGibbsstate(val_matH,vec_matH,invvec_matH,effpara0[1],effpara0[2])

    # local temperatures of global Gibbs state at bath L and R 
    dCgg = diag(Cgg)
    Egg_L = sum(dCgg[2:KL+1].*epsilonLR[2:KL+1]) # energy of bath L
    Egg_R = sum(dCgg[KL+2:end].*epsilonLR[KL+2:end]) # energy of bath R
    Ngg_L = sum(dCgg[2:KL+1]) # particle number of bath L
    Ngg_R = sum(dCgg[KL+2:end]) # particle number of bath R
    effparaL_gg = funeffectivebetamu(epsilonLR[2:KL+1],Egg_L,Ngg_L,betaL,muL) # local temperatures and chemical potential at bath L
    effparaR_gg = funeffectivebetamu(epsilonLR[KL+2:end],Egg_R,Ngg_R,betaR,muR) # at bath R
    println("betaL_gg=",effparaL_gg[1])
    println("muL_gg=",effparaL_gg[2])
    println("betaR_gg=",effparaR_gg[1])
    println("muR_gg=",effparaR_gg[2])

    # mutual info between S and E of global Gibbs state
    val_Cgg = real(eigvals(Cgg))
    vNEgg = vNEfrommatC(val_Cgg) # vNE for the total system
    val_Cgg_sys = Cgg[1,1]
    vNEgg_sys = - val_Cgg_sys.*log.(val_Cgg_sys) - (1.0 .- val_Cgg_sys).*log.(1.0 .- val_Cgg_sys) # vNE for the small system
    val_Cgg_E = real(eigvals(Cgg[2:end,2:end]))
    vNEgg_E = vNEfrommatC(val_Cgg_E) # vNE for the entire bath LR
    Igg_SE = vNEgg_sys + vNEgg_E - vNEgg # mutual info between S and E
    println("Igg_SE=",Igg_SE)

    # interbath correlation of global Gibbs state
    val_Cgg_L = real(eigvals(Cgg[2:KL+1,2:KL+1]))
    vNEgg_L = vNEfrommatC(val_Cgg_L) # vNE for bath L
    val_Cgg_R = real(eigvals(Cgg[KL+2:end,KL+2:end]))
    vNEgg_R = vNEfrommatC(val_Cgg_R) # vNE for bath R
    Igg_B = vNEgg_L + vNEgg_R - vNEgg_E # mutual info between L and R
    println("Igg_B=",Igg_B)

    # intrabath correlation of global Gibbs state
    diag_Cgg_L = real(diag(Cgg[2:KL+1,2:KL+1]))
    vNEgg_Lk = vNEfrommatC(diag_Cgg_L)
    Igg_L = vNEgg_Lk - vNEgg_L # intramode correlation in bath L
    diag_Cgg_R = real(diag(Cgg[KL+2:end,KL+2:end]))
    vNEgg_Rk = vNEfrommatC(diag_Cgg_R)
    Igg_R = vNEgg_Rk - vNEgg_R # intramode correlation in bath R
    println("Igg_L=",Igg_L)
    println("Igg_R=",Igg_R)

    # vNE of bath L and R at initial
    vNEpiL0 = compute_vNEpi(epsilonLR[2:KL+1],betaL,muL)
    vNEpiR0 = compute_vNEpi(epsilonLR[KL+2:end],betaR,muR)

    # define space for computing the dynamics
    Ct = zeros(ComplexF64,KL+KR+1,KL+KR+1)
    dCt = zeros(ComplexF64,KL+KR+1)
    dCt1 = zeros(ComplexF64,KL+KR+1)
    val_Ct = zeros(Float64,KL+KR+1)
    val_Ct_E = zeros(Float64,KL+KR)
    diag_Ct_E = zeros(Float64,KL+KR)
    val_Ct_L = zeros(Float64,KL)
    val_Ct_R = zeros(Float64,KR)

    Ct_saved = zeros(ComplexF64,KL+KR+1,KL+KR+1,Nt)
    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)
    QL = zeros(ComplexF64,Nt)
    QR = zeros(ComplexF64,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    vNE_sys = zeros(Float64,Nt)
    vNE_E = zeros(Float64,Nt)
    vNE_L = zeros(Float64,Nt)
    vNE_R = zeros(Float64,Nt)
    vNE_Lk = zeros(Float64,Nt)
    vNE_Rk = zeros(Float64,Nt)
    vNE = zeros(Float64,Nt)
    vNEpiL = zeros(Float64,Nt)
    vNEpiR = zeros(Float64,Nt)

    I_SE = zeros(ComplexF64,Nt)
    I_B = zeros(ComplexF64,Nt)
    I_L = zeros(ComplexF64,Nt)
    I_R = zeros(ComplexF64,Nt)
    Denv = zeros(ComplexF64,Nt)
    Denveq = zeros(ComplexF64,Nt)
    Drel_rhoL_piL_ratio = zeros(Float64,Nt)
    Drel_rhoR_piR_ratio = zeros(Float64,Nt)
    
    sigma_c = zeros(ComplexF64,Nt)
    sigma_d = zeros(ComplexF64,Nt)

    for tt = 1:Nt

        # time evolution
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # energy
        dCt .= diag(Ct)
        E_sys[tt] = dCt[1]*epsilonLR[1] # small system
        E_L[tt] = sum(dCt[2:KL+1].*epsilonLR[2:KL+1]) # bath L
        E_R[tt] = sum(dCt[KL+2:end].*epsilonLR[KL+2:end]) # bath R
        dCt1 .= Ct[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2) # total system

        # particle numbers
        N_sys[tt] = dCt[1] # small system
        N_L[tt] = sum(dCt[2:KL+1]) # bath L
        N_R[tt] = sum(dCt[KL+2:end]) # bath R
        
        # I_SE
        val_Ct .= real(eigvals(Ct))
        vNE[tt] = vNEfrommatC(val_Ct) # vNE of the total system
        vNE_sys[tt] = vNEfrommatC(real(Ct[1,1])) # vNE of the small system
        val_Ct_E .= real(eigvals(Ct[2:end,2:end]))
        vNE_E[tt] = vNEfrommatC(val_Ct_E) # vNE of the entire bath LR
        I_SE[tt] = vNE_sys[tt] - vNE_sys[1] + vNE_E[tt] - vNE_E[1]

        # I_B
        val_Ct_L .= real(eigvals(Ct[2:KL+1,2:KL+1]))
        vNE_L[tt] = vNEfrommatC(val_Ct_L) # vNE of bath L
        val_Ct_R .= real(eigvals(Ct[KL+2:end,KL+2:end]))
        vNE_R[tt] = vNEfrommatC(val_Ct_R) # vNE of bath R
        I_B[tt] = vNE_L[tt] + vNE_R[tt] - vNE_E[tt]

        # I_L and I_R
        diag_Ct_E .= real(diag(Ct[2:end,2:end]))
        vNE_Lk[tt] = vNEfrommatC(diag_Ct_E[1:KL])
        I_L[tt] = vNE_Lk[tt] - vNE_L[tt]
        vNE_Rk[tt] = vNEfrommatC(diag_Ct_E[KL+1:end])
        I_R[tt] = vNE_Rk[tt] - vNE_R[tt]

        # effective inverse temperature and chemical potential
        if tt == 1
           betaL0 = betaL
           betaR0 = betaR
           muL0 = muL
           muR0 = muR
        else # tt > 1
           betaL0 = effparaL[tt-1,1]
           betaR0 = effparaR[tt-1,1]
           muL0 = effparaL[tt-1,2]
           muR0 = effparaR[tt-1,2]
        end
        effparaL[tt,:] .= funeffectivebetamu(epsilonLR[2:KL+1],real(E_L[tt]),real(N_L[tt]),betaL0,muL0) # bath L
        effparaR[tt,:] .= funeffectivebetamu(epsilonLR[KL+2:end],real(E_R[tt]),real(N_R[tt]),betaR0,muR0) # bath R

        # heat
        dCt .= diag(Ct - C0)
        QL[tt] = -sum(dCt[2:KL+1].*(epsilonLR[2:KL+1] .- muL))
        QR[tt] = -sum(dCt[KL+2:end].*(epsilonLR[KL+2:end] .- muR))
        
        # vNE of themal state at effparaL and effparaR at bath L and R 
        vNEpiL[tt] = compute_vNEpi(epsilonLR[2:KL+1],effparaL[tt,1],effparaL[tt,2])
        vNEpiR[tt] = compute_vNEpi(epsilonLR[KL+2:end],effparaR[tt,1],effparaR[tt,2])

        # entropy production
        sigma_c[tt] = vNE_sys[tt] - vNE_sys[1] + (vNEpiL[tt]-vNEpiL0) + (vNEpiR[tt]-vNEpiR0) # capital sigma defined in Eq. (7)
        sigma_d[tt] = vNE_sys[tt] - vNE_sys[1] - betaL*QL[tt] - betaR*QR[tt] # lowercase sigma defined in Eq. (13)
        
        # local athermality
        Denv[tt] =  sigma_c[tt] - I_SE[tt] - I_B[tt] - I_L[tt] - I_R[tt]
        Denveq[tt] = sigma_d[tt] - sigma_c[tt]

        # relative entropy between rho_nu(t) and pi_nu(t)
        # ratio with the bound
        Drel_rhoL_piL_ratio[tt] = (-vNE_L[tt] + vNEpiL[tt])/vNEpiL[tt]
        Drel_rhoR_piR_ratio[tt] = (-vNE_R[tt] + vNEpiR[tt])/vNEpiR[tt]

        # save the correlation matrix
        Ct_saved[:,:,tt] = Ct

        println(tt)

    end

    return time, E_L, E_R, E_tot, N_L, N_R, N_sys, QL, QR, effpara0, effparaL, effparaR, Drel_rhoL_piL_ratio, Drel_rhoR_piR_ratio, sigma_c, sigma_d, I_SE, I_B, I_L, I_R, Denv, Denveq, Ct_saved
    
end

# plot ##################

# Figures 3 and 7
function plot_initialstate(K::Int64,W::Int64,beta::Float64,mu::Float64,pure::Int64)

    # energy level
    epsilon = zeros(Float64,K)
    Depsilon = W/(K-1)
    for kk = 1:K
        epsilon[kk] = (kk-1)*Depsilon - W/2
    end

    # Fermi distribution
    nF = zeros(Float64,K)
    for kk = 1:K
        nF[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
    end

    plot(epsilon,nF,lw=4,color=:gray,framestyle = :box)

    # random pure state
    if pure == 1
       seed = 5
       Random.seed!(seed)
       npure = rand(Uniform(0,1), K)
       for kk = 1:K
           if npure[kk] <= nF[kk]
              npure[kk] = 1
           else #npure[kk] > nF[kk]
              npure[kk] = 0
           end
       end
       plot!(epsilon,npure,lw=0.5,color=:black,marker=(:circle,4))
    end

    ylims!((-0.1,1.1))
    plot!(legend=:none)

end

# Figures 4 and 8
function plot_efftem(effparaL,effparaR,effpara0,time)

    Nt = length(time)

    plot(log10.(time),real(effparaL[:,1]),lw=4,label=L"\beta_{L,t}^*",palette=:reds,framestyle = :box)
    plot!(log10.(time),real(effparaR[:,1]),lw=4,label=L"\beta_{R,t}^*",palette=:reds)
    
    plot!(log10.(time),real(effpara0[1]*ones(Nt)),lw=2,color=:black,ls=:dash,label=L"\beta_{ref}^*")

    plot!([-1,-1.000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime I")
    plot!([5,5.0000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime II")

    ylims!((0,11))
    xlims!((-4.5,8.5))

    plot!(xlabel=L"log_{10} t")
    # plot!(aspect_ratio=6.0)
    plot!(legend=:none)

end

# Figures 4 and 8
function plot_effchem(effparaL,effparaR,effpara0,time)

    Nt = length(time)

    plot(log10.(time),real(effparaL[:,2]),lw=4,label=L"\mu_{L,t}^*",palette=:blues,framestyle = :box)
    plot!(log10.(time),real(effparaR[:,2]),lw=4,label=L"\mu_{R,t}^*",palette=:blues)

    plot!(log10.(time),real(effpara0[2]*ones(Nt)),lw=2,color=:black,ls=:dash,label=L"\mu_{ref}^*")

    plot!([-1,-1.000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime I")
    plot!([5,5.0000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime II")

    ylims!((-1.1,1.1))
    xlims!((-4.5,8.5))
    
    plot!(xlabel=L"log_{10} t")
    # plot!(aspect_ratio=3.0)
    plot!(legend=:none)

end

# Figure 4
function plot_Drelratio(Drel_rhoL_piL_ratio,Drel_rhoR_piR_ratio,time)

    plot(log10.(time),Drel_rhoL_piL_ratio,lw=4,label=L"L",palette=:greens,framestyle = :box)
    plot!(log10.(time),Drel_rhoR_piR_ratio,lw=4,label=L"R",palette=:greens,framestyle = :box)

    plot!([-1,-1.000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime I")
    plot!([5,5.0000001],[-1,100],lw=2,color=:black,ls=:dot,label=L"regime II")

    xlims!((-4.5,8.5))
    ylims!((-0.1,1.1))
    # plot!(legend=:none)

    plot!(xlabel=L"log_{10} t")

end

# Figure 5
function plot_sigmas(sigma_d,sigma_c,I_SE,I_B,I_L,I_R,time)

    plot(log10.(time[2:end]),real(log.(sigma_d[2:end])),label=L"\sigma",color=:grey,lw=10,framestyle = :box)
    plot!(log10.(time[2:end]),real(log.(sigma_c[2:end])),label=L"\Sigma",color=:black,lw=9)
    plot!(log10.(time[2:end]),real(log.(I_SE[2:end]+I_L[2:end]+I_R[2:end]+I_B[2:end])),label=L"I_{SB}+I_{B}+I_L+I_R",color=:red,lw=6,ls=:dot)
    plot!(log10.(time[2:end]),real(log.(I_SE[2:end]+I_L[2:end]+I_R[2:end])),label=L"I_{SB}+I_{B}",lw=3,color=:blue,ls=:dash)
    plot!(log10.(time[2:end]),real(log.(I_SE[2:end])),label=L"I_{SB}",lw=2,color=:green)

    plot!([-1,-1.000001],[-100,100],lw=2,color=:black,ls=:dot,label=L"regime I")
    plot!([5,5.0000001],[-100,100],lw=2,color=:black,ls=:dot,label=L"regime II")

    plot!(legend=:none)
    xlims!((-4.5,8.5))
    ylims!((-4,9))

    # plot!(aspect_ratio=14.0/10)

end

# Figure 5
function plot_sigmas_sub(I_SE,I_B,I_L,I_R,time)

    plot(log10.(time[6:end]),log.(real(I_B[6:end])),label=L"I_{B}",lw=9, color=RGB(51/255,160/255,44/255))
    plot!(log10.(time[2:end]),log.(real(I_L[2:end])),label=L"I_{L}",lw=7, color=RGB(178/255,223/255,138/255))
    plot!(log10.(time[2:end]),log.(real(I_R[2:end])),label=L"I_{R}",lw=4, color=RGB(31/255,120/255,180/255))
    plot!(log10.(time[2:end]),log.(real(I_SE[2:end])),label=L"I_{SB}",lw=2,framestyle = :box, color=RGB(166/255,206/255,227/255))
    
    plot!([-1,-1.000001],[-100,100],lw=2,color=:black,ls=:dot,label=L"regime I")
    plot!([5,5.0000001],[-100,100],lw=2,color=:black,ls=:dot,label=L"regime II")

    plot!(legend=:none)
    xlims!((-4.5,8.5))
    ylims!((-4,9))
    
    # plot!(aspect_ratio=14.0/10)

end

# Figure 6
function averagecorrelationsregimeIII(KL::Int64,KR::Int64,W::Float64,betaL::Float64,betaR::Float64,muL::Float64,muR::Float64)

    array_Gamma = [10.0^(0.5)]
    # array_Gamma = [10.0^(0.5), 10.0^(1), 10.0^(1.5), 10.0^(2), 10.0^(2.5), 10.0^(3)]

    tt_ref0 = 10.0^5
    tt_ref1 = 10.0^8 

    array_I_SE = zeros(Float64,length(array_Gamma))
    array_I_B = zeros(Float64,length(array_Gamma))
    array_I_L = zeros(Float64,length(array_Gamma))
    array_I_R = zeros(Float64,length(array_Gamma))
    array_Denv = zeros(Float64,length(array_Gamma))
    array_Denveq = zeros(Float64,length(array_Gamma))

    for jj = 1:length(array_Gamma)

        Gamma = array_Gamma[jj]
        time, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, I_SE, I_B, I_L, I_R, Denv, Denveq, ~ = dynamics(0.0,KL,KR,W,betaL,betaR,Gamma,Gamma,muL,muR,tt_ref0,tt_ref1,22,0)
        
        # Consider time[1] = 0.0, time[2] = tt_ref0, time[end] = tt_ref1
        array_I_SE[jj] = mean(real(I_SE[2:end])) 
        array_I_B[jj] = mean(real(I_B[2:end]))
        array_I_L[jj] = mean(real(I_L[2:end]))
        array_I_R[jj] = mean(real(I_R[2:end]))
        array_Denv[jj] = mean(real(Denv[2:end]))
        array_Denveq[jj] = mean(real(Denveq[2:end]))

    end

    return array_Gamma, array_I_SE, array_I_B, array_I_L, array_I_R, array_Denv, array_Denveq

end

# Figure 6
function plot_averagecorrelationsregimeIII_each(array_Gamma, array_I_SE, array_I_B, array_I_L, array_I_R, array_Denv, array_Denveq)

    plot(log10.(array_Gamma),log.(array_I_SE),marker=(:circle,8),lw=3,label=L"\langle I_{SB} \rangle", framestyle = :box)

    plot!(log10.(array_Gamma),log.(array_I_B),marker=(:square,8),lw=3,label=L"\langle I_{B} \rangle")
    
    plot!(log10.(array_Gamma),log.(array_I_L),marker=(:utriangle,8),lw=3,label=L"\langle I_{L} \rangle")
    
    plot!(log10.(array_Gamma),log.(array_I_R),marker=(:dtriangle,8),lw=3,label=L"\langle I_{R} \rangle")
    
    plot!(log10.(array_Gamma),log.(array_Denv),marker=(:pentagon,8),lw=3,label=L"\langle D_{env} \rangle")

    plot!(log10.(array_Gamma),log.(array_Denveq),marker=(:diamond,10),lw=3,label=L"\langle \tilde{D}_{env} \rangle")

    xlims!((0,3.6))
    ylims!((-0.5,8.1))
    plot!(aspect_ratio=3.6/8.6)
    # plot!(legend=:none)

end