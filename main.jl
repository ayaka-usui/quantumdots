
using Arpack, SparseArrays, LinearAlgebra
# using ExpmV
using NLsolve
using Plots
using Distributions, Random
using JLD
using Combinatorics

function createH!(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
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

function funbetamu!(F,x,epsilon::Vector{Float64},K::Int64,W::Int64,Ene::Float64,Np::Float64)

    # x[1] = beta, x[2] = mu
    # Depsilon = W/(K-1)

    for kk = 1:K
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

function funeffectivebetamu(K::Int64,W::Int64,epsilon::Vector{Float64},Ene::Float64,Np::Float64,beta0::Float64,mu0::Float64)

    sol = nlsolve((F,x) ->funbetamu!(F,x,epsilon,K,W,Ene,Np), [beta0; mu0])
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

function calculateptotalobsvNE_test2(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

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

    pop = zeros(Float64,2^5)
    pop[1] = C0[1]*C0[2]*C0[3]*C0[4]*C0[5]
    pop[2] = C0[1]*C0[2]*C0[3]*C0[4]*(1-C0[5])
    pop[3] = C0[1]*C0[2]*C0[3]*(1-C0[4])*C0[5]
    pop[4] = C0[1]*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])

    pop[5] = C0[1]*C0[2]*(1-C0[3])*C0[4]*C0[5]
    pop[6] = C0[1]*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])
    pop[7] = C0[1]*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]
    pop[8] = C0[1]*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])

    pop[9] = C0[1]*(1-C0[2])*C0[3]*C0[4]*C0[5]
    pop[10] = C0[1]*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])
    pop[11] = C0[1]*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]
    pop[12] = C0[1]*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])
    pop[13] = C0[1]*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]
    pop[14] = C0[1]*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])
    pop[15] = C0[1]*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]
    pop[16] = C0[1]*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])

    pop[17] = (1-C0[1])*C0[2]*C0[3]*C0[4]*C0[5]
    pop[18]= (1-C0[1])*C0[2]*C0[3]*C0[4]*(1-C0[5])
    pop[19]= (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*C0[5]
    pop[20]= (1-C0[1])*C0[2]*C0[3]*(1-C0[4])*(1-C0[5])
    pop[21]= (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*C0[5]
    pop[22]= (1-C0[1])*C0[2]*(1-C0[3])*C0[4]*(1-C0[5])
    pop[23]= (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*C0[5]
    pop[24]= (1-C0[1])*C0[2]*(1-C0[3])*(1-C0[4])*(1-C0[5])
    pop[25]= (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*C0[5]
    pop[26]= (1-C0[1])*(1-C0[2])*C0[3]*C0[4]*(1-C0[5])
    pop[27]= (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*C0[5]
    pop[28]= (1-C0[1])*(1-C0[2])*C0[3]*(1-C0[4])*(1-C0[5])
    pop[29] = (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*C0[5]
    pop[30] = (1-C0[1])*(1-C0[2])*(1-C0[3])*C0[4]*(1-C0[5])
    pop[31] = (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*C0[5]
    pop[32] = (1-C0[1])*(1-C0[2])*(1-C0[3])*(1-C0[4])*(1-C0[5])

    # test0 = zeros(Float64,1,5)
    # test0[1,3] = ptotal0
    # ptotal = [test0;ptotal]
    # ptotal = ptotal .+ 1e-15

    Sobs = -sum(pop.*log.(pop))
    println(Sobs)

    # vNE
    vNE = - sum(C0.*log.(C0)) - sum((1.0 .- C0).*log.(1.0 .- C0))
    println(vNE)

    return pop
    # return Sobs

end

function calculateptotaltime_test2(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

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
    Ct = zeros(ComplexF64,2*K+1,2*K+1)
    eigval_Ct = zeros(Float64,2*K+1)
    eigvec_Ct = zeros(Float64,2*K+1,2*K+1)

    ptotal = zeros(Float64,2^(2*K+1))
    diagC0 = C0
    C0 = diagm(C0)

    for tt = 1:Nt

        # time evolution of correlation matrix
        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH

        # total
        lambda, eigvec_Ct = eigen(Ct)
        eigval_Ct .= real.(lambda)
        # eigval_Ct .= real.(diag(Ct))

        # nj = (abs.(eigvec_Ct).^2)*eigval_Ct
        # eigval_Ct .= nj

        ptotal .= 0.0

        ptotal[1] = eigval_Ct[1]*eigval_Ct[2]*eigval_Ct[3]*eigval_Ct[4]*eigval_Ct[5]
        ptotal[2] = eigval_Ct[1]*eigval_Ct[2]*eigval_Ct[3]*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[3] = eigval_Ct[1]*eigval_Ct[2]*eigval_Ct[3]*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[4] = eigval_Ct[1]*eigval_Ct[2]*eigval_Ct[3]*(1-eigval_Ct[4])*(1-eigval_Ct[5])

        ptotal[5] = eigval_Ct[1]*eigval_Ct[2]*(1-eigval_Ct[3])*eigval_Ct[4]*eigval_Ct[5]
        ptotal[6] = eigval_Ct[1]*eigval_Ct[2]*(1-eigval_Ct[3])*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[7] = eigval_Ct[1]*eigval_Ct[2]*(1-eigval_Ct[3])*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[8] = eigval_Ct[1]*eigval_Ct[2]*(1-eigval_Ct[3])*(1-eigval_Ct[4])*(1-eigval_Ct[5])

        ptotal[9] = eigval_Ct[1]*(1-eigval_Ct[2])*eigval_Ct[3]*eigval_Ct[4]*eigval_Ct[5]
        ptotal[10] = eigval_Ct[1]*(1-eigval_Ct[2])*eigval_Ct[3]*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[11] = eigval_Ct[1]*(1-eigval_Ct[2])*eigval_Ct[3]*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[12] = eigval_Ct[1]*(1-eigval_Ct[2])*eigval_Ct[3]*(1-eigval_Ct[4])*(1-eigval_Ct[5])
        ptotal[13] = eigval_Ct[1]*(1-eigval_Ct[2])*(1-eigval_Ct[3])*eigval_Ct[4]*eigval_Ct[5]
        ptotal[14] = eigval_Ct[1]*(1-eigval_Ct[2])*(1-eigval_Ct[3])*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[15] = eigval_Ct[1]*(1-eigval_Ct[2])*(1-eigval_Ct[3])*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[16] = eigval_Ct[1]*(1-eigval_Ct[2])*(1-eigval_Ct[3])*(1-eigval_Ct[4])*(1-eigval_Ct[5])

        ptotal[17] = (1-eigval_Ct[1])*eigval_Ct[2]*eigval_Ct[3]*eigval_Ct[4]*eigval_Ct[5]
        ptotal[18]= (1-eigval_Ct[1])*eigval_Ct[2]*eigval_Ct[3]*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[19]= (1-eigval_Ct[1])*eigval_Ct[2]*eigval_Ct[3]*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[20]= (1-eigval_Ct[1])*eigval_Ct[2]*eigval_Ct[3]*(1-eigval_Ct[4])*(1-eigval_Ct[5])
        ptotal[21]= (1-eigval_Ct[1])*eigval_Ct[2]*(1-eigval_Ct[3])*eigval_Ct[4]*eigval_Ct[5]
        ptotal[22]= (1-eigval_Ct[1])*eigval_Ct[2]*(1-eigval_Ct[3])*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[23]= (1-eigval_Ct[1])*eigval_Ct[2]*(1-eigval_Ct[3])*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[24]= (1-eigval_Ct[1])*eigval_Ct[2]*(1-eigval_Ct[3])*(1-eigval_Ct[4])*(1-eigval_Ct[5])
        ptotal[25]= (1-eigval_Ct[1])*(1-eigval_Ct[2])*eigval_Ct[3]*eigval_Ct[4]*eigval_Ct[5]
        ptotal[26]= (1-eigval_Ct[1])*(1-eigval_Ct[2])*eigval_Ct[3]*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[27]= (1-eigval_Ct[1])*(1-eigval_Ct[2])*eigval_Ct[3]*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[28]= (1-eigval_Ct[1])*(1-eigval_Ct[2])*eigval_Ct[3]*(1-eigval_Ct[4])*(1-eigval_Ct[5])
        ptotal[29] = (1-eigval_Ct[1])*(1-eigval_Ct[2])*(1-eigval_Ct[3])*eigval_Ct[4]*eigval_Ct[5]
        ptotal[30] = (1-eigval_Ct[1])*(1-eigval_Ct[2])*(1-eigval_Ct[3])*eigval_Ct[4]*(1-eigval_Ct[5])
        ptotal[31] = (1-eigval_Ct[1])*(1-eigval_Ct[2])*(1-eigval_Ct[3])*(1-eigval_Ct[4])*eigval_Ct[5]
        ptotal[32] = (1-eigval_Ct[1])*(1-eigval_Ct[2])*(1-eigval_Ct[3])*(1-eigval_Ct[4])*(1-eigval_Ct[5])


        Sobs = -sum(ptotal.*log.(ptotal))
        println(Sobs)

        # vNE
        vNE = - sum(diagC0.*log.(diagC0)) - sum((1.0 .- diagC0).*log.(1.0 .- diagC0))
        println(vNE)

        vNE_sys = -Ct[1,1]*log(Ct[1,1]) - (1-Ct[1,1])*log(1-Ct[1,1])
        println(vNE_sys)

    end

    # return SobsL, SobsR

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

function calculatequantities2(K::Int64,W::Int64,numvari::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    # Hamiltonian
    matH = spzeros(Float64,K*2+1,K*2+1)
    createH_Deltaepsilon!(K,W,numvari,betaL,betaR,GammaL,GammaR,matH)

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
    val_Ct = zeros(ComplexF64,K*2+1)
    dCt1 = zeros(ComplexF64,K*2+1)

    epsilonLR = diag(matH)
    tLRk = matH[1,1:end]

    vNE_sys = zeros(ComplexF64,Nt)
    vNE_E = zeros(ComplexF64,Nt)
    vNE_alphak = zeros(ComplexF64,Nt)
    I_SE = zeros(ComplexF64,Nt)
    betaQL = zeros(ComplexF64,Nt)
    betaQR = zeros(ComplexF64,Nt)
    sigma = zeros(ComplexF64,Nt)
    Drel = zeros(ComplexF64,Nt)
    I_env = zeros(ComplexF64,Nt)

    E_sys = zeros(ComplexF64,Nt)
    E_L = zeros(ComplexF64,Nt)
    E_R = zeros(ComplexF64,Nt)
    F_L = zeros(ComplexF64,Nt)
    F_R = zeros(ComplexF64,Nt)
    E_tot = zeros(ComplexF64,Nt)
    vNE = zeros(ComplexF64,Nt)
    N_sys = zeros(ComplexF64,Nt)
    N_L = zeros(ComplexF64,Nt)
    N_R = zeros(ComplexF64,Nt)

    effparaL = zeros(Float64,Nt,2)
    effparaR = zeros(Float64,Nt,2)

    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH
        # Ct = Hermitian(Ct)

        # energy
        dCt .= diag(Ct) #diag(Ct - C0)
        E_sys[tt] = dCt[1]*epsilonLR[1]
        E_L[tt] = sum(dCt[2:K+1].*epsilonLR[2:K+1])
        E_R[tt] = sum(dCt[K+2:2*K+1].*epsilonLR[K+2:2*K+1])

        dCt1 .= Ct[1,1:end] # Ct[1,1:end] - C0[1,1:end]
        E_tot[tt] = E_L[tt] + E_R[tt] + sum(dCt1[2:end].*tLRk[2:end])*2

        # particle numbers
        N_sys[tt] = dCt[1]
        N_L[tt] = sum(dCt[2:K+1])
        N_R[tt] = sum(dCt[K+2:2*K+1])

        # vNE for total
        val_Ct .= eigvals(Ct)
        vNE[tt] = - sum(val_Ct.*log.(val_Ct)) - sum((1.0 .- val_Ct).*log.(1.0 .- val_Ct))

        # println(vNE[tt])

        # effective inverse temperature and chemical potential
        effparaL[tt,:] .= funeffectivebetamu(K,W,epsilonLR[2:K+1],real(E_L[tt]),real(N_L[tt]),betaL,muL)
        effparaR[tt,:] .= funeffectivebetamu(K,W,epsilonLR[K+2:2*K+1],real(E_R[tt]),real(N_R[tt]),betaR,muR)

        # effparaL[tt,:] .= funeffectivebetamu(K,W,real(E_L[tt]),real(N_L[tt]),betaL,muL)
        # effparaR[tt,:] .= funeffectivebetamu(K,W,real(E_R[tt]),real(N_R[tt]),betaR,muR)

    end

    return time, E_sys, E_L, E_R, N_sys, N_L, N_R, E_tot, effparaL, effparaR

end

# println("add fluctuations to tunnelling coupling")
# Depsilon = W/(K-1)
# tunnelL = sqrt(GammaL*Depsilon/(2*pi))
# tunnelR = sqrt(GammaR*Depsilon/(2*pi))
# for kk = 1:K
#     flucutu = tunnelL*rand(Uniform(-1,1))/20
#     matH[1+kk,1] = tunnelL + flucutu  # tunnel with the bath L
#     flucutu = tunnelR*rand(Uniform(-1,1))/20
#     matH[1+K+kk,1] = tunnelR + flucutu # tunnel with the bath R
# end

function calculatequantities(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

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
