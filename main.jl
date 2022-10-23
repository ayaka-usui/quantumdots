
using Arpack, SparseArrays, LinearAlgebra
# using ExpmV
using NLsolve
using Plots
using Distributions, Random
using JLD
using Combinatorics

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

function funbetamu2!(F,x,epsilon::Vector{Float64},Ene::Float64,Np::Float64,Cgg::Vector{Float64},matCgg::Matrix{Float64},K::Int64,val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64})

    # x[1] = beta, x[2] = mu

    # Cgg = zeros(Float64,K*2+1)
    Cgg .= 0.0
    for kk = 1:2*K+1
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

function funeffectivebetamu2(epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64,Cgg::Vector{Float64},matCgg::Matrix{Float64},K::Int64,val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64})

    sol = nlsolve((F,x) ->funbetamu2!(F,x,epsilon,Ene,Np,Cgg,matCgg,K,val_matH,vec_matH,invvec_matH), [beta0; mu0])
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

function globalGibbsstate(K::Int64,val_matH::Vector{Float64},vec_matH::Matrix{Float64},invvec_matH::Matrix{Float64},beta::Float64,mu::Float64)

    # global Gibbs state
    Cgg = zeros(Float64,K*2+1)
    for kk = 1:2*K+1
        Cgg[kk] = 1.0/(exp((val_matH[kk]-mu)*beta)+1.0)
    end
    Cgg = diagm(Cgg)
    Cgg .= vec_matH*Cgg*invvec_matH

    return Cgg

end

function heatcapacityeff(C0::Vector{Float64},K::Int64,epsilon::Vector{Float64},beta::Float64,mu::Float64)

    # C0 = zeros(Float64,K)
    C0 .= 0.0
    for kk = 1:K
        C0[kk] = 1.0/(exp((epsilon[kk]-mu)*beta)+1.0)
    end

    varH = sum(epsilon.^2 .*(C0.*(1.0.-C0)))
    varN = sum(C0.*(1.0.-C0))
    varHN = sum(epsilon.*(C0.*(1.0.-C0)))

    dUdbeta = -varH + mu*varHN
    dUdmu = beta*varHN
    dNdbeta = mu*varN - varHN
    dNdmu = beta*varN

    return [dUdbeta dUdmu; dNdbeta dNdmu]

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
    effpara0 = funeffectivebetamu2(epsilonLR,E_tot0,N_tot0,(betaL+betaR)/2,(muL+muR)/2,Cgg0,matCgg0,K,val_matH,vec_matH,invvec_matH)
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

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + real(sum(dCt1[2:end].*tLRk[2:end])*2)

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:K+1])
        N_R[tt] = sum(dCt[K+2:2*K+1])

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

    return time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR
    # return time, vNE_sys, effparaL, effparaR, QL, QR
    # return time, sigma, sigma3, sigma_c, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel
    # return time, sigma, sigma2, sigma3, sigma_c
    # return time, betaQL, betaQLtime, betaQR, betaQRtime
    # return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

# save("data_calculatequantities2_K128W20betaL1R05GammaL05R05muL1R1tf1000Nt10001.jld", "time", time, "sigma", sigma, "sigma3", sigma3, "sigma_c", sigma_c, "effparaL", effparaL, "effparaR", effparaR, "I_SE", I_SE, "I_B", I_B, "I_L", I_L, "I_R", I_R, "I_env", I_env, "Drel", Drel)

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

function plot_sigmas(time,GammaLR,sigma,I_SE,I_B,I_L,I_R,Drelnuk)

    I_SE_mvave = movingmean(real(I_SE),1001);
    I_L_mvave = movingmean(real(I_L),1001);
    I_R_mvave = movingmean(real(I_R),1001);

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
    end

    p1 = plot(log10.(time*GammaLR),log10.(real(sigma)),color=:black,lw=3,label=L"\sigma")
    plot!(log10.(time*GammaLR),log10.(real(I_SE)),color=:red,lw=3,label=L"I_{SB}")
    plot!(log10.(time*GammaLR),log10.(real(I_B)),color=:blue,lw=3,label=L"I_{B}")
    plot!(log10.(time*GammaLR),log10.(real(I_L)),color=:green,lw=3,label=L"I_{L}")
    plot!(log10.(time*GammaLR),log10.(real(I_R)),color=:orange,lw=3,label=L"I_{R}")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(Drelnuk[2:end])),color=:purple,lw=3,label=L"D_{env}")

    plot!(log10.(time_some*GammaLR),log10.(real(I_SE_some)),color=:red,lw=0,markershape=:circle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_B_some)),color=:blue,lw=0,markershape=:rect,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_L_some)),color=:green,lw=0,markershape=:utriangle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_R_some)),color=:orange,lw=0,markershape=:dtriangle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(Drelnuk_some)),color=:purple,lw=0,markershape=:pentagon,ms=6)

    xlims!((-1.1,0.7))
    ylims!((-3,3))
    plot!(legend=:none)

    p2 = plot(log10.(time*GammaLR),log10.(real(sigma)),color=:black,lw=3,label=L"\sigma")
    plot!(log10.(time*GammaLR),log10.(real(I_SE_mvave)),color=:red,lw=3,ls=:dash,label=L"I_{SB}")
    plot!(log10.(time*GammaLR),log10.(real(I_B)),color=:blue,lw=3,label=L"I_{B}")
    plot!(log10.(time*GammaLR),log10.(real(I_L_mvave)),color=:green,lw=3,ls=:dash,label=L"I_{L}")
    plot!(log10.(time*GammaLR),log10.(real(I_R_mvave)),color=:orange,lw=3,ls=:dash,label=L"I_{R}")
    plot!(log10.(time[2:end]*GammaLR),log10.(real(Drelnuk[2:end])),color=:purple,lw=3,label=L"D_{env}")

    plot!(log10.(time_some*GammaLR),log10.(real(I_SE_mvave_some)),color=:red,lw=0,markershape=:circle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_B_some)),color=:blue,lw=0,markershape=:rect,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_L_mvave_some)),color=:green,lw=0,markershape=:utriangle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(I_R_mvave_some)),color=:orange,lw=0,markershape=:dtriangle,ms=6)
    plot!(log10.(time_some*GammaLR),log10.(real(Drelnuk_some)),color=:purple,lw=0,markershape=:pentagon,ms=6)

    xlims!((0.5,4.0))
    ylims!((-3,3))
    plot!(legend=:none)

    plot(p1,p2,layout=(1,2),size=(700,400))

    # plot(p1,layout=(1,1),size=(700,400))

end

function averagecorrelationsregimeIII(K::Int64,W::Int64,t_flu::Float64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    array_Gamma = [0.1, sqrt(0.1), 1.0, sqrt(10.0), 10.0, 10.0^2]
    array_tt0 = zeros(Int64,length(array_Gamma))
    array_I_SE = zeros(Float64,length(array_Gamma))
    array_I_B = zeros(Float64,length(array_Gamma))
    array_I_L = zeros(Float64,length(array_Gamma))
    array_I_R = zeros(Float64,length(array_Gamma))

    for jj = 1:length(array_Gamma)

        Gamma = array_Gamma[jj]

        if jj != length(array_Gamma)
           time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR = calculatequantities2(K,W,t_flu,betaL,betaR,Gamma,Gamma,muL,muR,50000.0,501)

           tt0 = argmin(abs.(time*Gamma.-10^3))
           if time[tt0] < 10^3
              tt0 = tt0 + 1
           end

        else

           time, sigma, sigma2, sigma3, sigma_c, effpara0, effparaL, effparaR, I_SE, I_B, I_L, I_R, I_env, Drel, Drelnuk, Drelpinuk, betaQL, betaQR, betaQLtime, betaQRtime, dQLdt, dQRdt, matCL, matCR = calculatequantities2(K,W,t_flu,betaL,betaR,Gamma,Gamma,muL,muR,500.0,501)
           tt0 = argmin(abs.(time*Gamma.-10^4))
           if time[tt0] < 10^4
              tt0 = tt0 + 1
           end
        end

        array_tt0[jj] = tt0
        array_I_SE[jj] = mean(real(I_SE[tt0:end]))
        array_I_B[jj] = mean(real(I_B[tt0:end]))
        array_I_L[jj] = mean(real(I_L[tt0:end]))
        array_I_R[jj] = mean(real(I_R[tt0:end]))

    end

    return array_Gamma, array_tt0, array_I_SE, array_I_B, array_I_L, array_I_R

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
