---
layout: article
title: RNA-RNA interaction
mathjax: true
tags: Papers
---

# Protein-RNA interaction

## I Computational problems for protein-RNA interaction

Do you prefer to know more information about the interaction from the **Protein side** or  **RNA side**?

### 1.1 Binding site prediction (protein side)

#### Target:

**Predict the binding sites and binding positions on the protein surface for RNAs.**

1. Given a protein, judge whether this **protein is an RBP** or not.

2. Which **AAs on the protein** sequence can potentially**interact with RNAs**

#### Step: (do not use the RNA information)

1. Input: a protein (sequence information or the structure information, or both)

2. Extract protein features or define certain scoring functions

3. Model: machine learning model or an alignmentbased method

4. Output: (binary predictions) 
   
   * **Site-level-based**
   
   * **Protein-level-based**

**The methods based on structure have better performance on this problem than the sequence-based methods.**

### 1.2 Binding preference predition (RNA side)

#### Target:

**Predict the binding preference of an RNA-binding protein (RBP)**

1. Given a RBP, judge which RNAs can interact with

#### Step:  (do not use the protein information)

1. **Input:** a set of RNA sequence (sequence information or the structure information, or both)

2. Extract RNA features

3. **Model:** machine learning model or a statistical motif model

4. **Output:** RNA interaction probability (whether RNA can interact with the protein)

**Both RNA information and Protein information can be used directly to predict the interaction perference.**

## II Datasets

![1.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/17-12-26-24-1.png)

> [1] [Protein–RNA interaction prediction with deep learning: structure matters](https://academic.oup.com/bib/article/23/1/bbab540/6470965)

# RNA-RNA interaction

## I Base

### 1.1 Understanding the RNA

- Like DNA, RNA is assembled as a **chain of nucleotides**, but unlike DNA it is more often found in nature as a **single-strand folded onto itself**, rather than a paired double-strand.

- Type:
  
  - **messenger RNA (mRNA)**:catalyzing biological reactions, controlling gene expression, or sensing and communicating responses to cellular signals.
  
  - **transfer RNA (tRNA)**: deliver amino acids to the ribosome
  
  - **ribosomal RNA (rRNA)**:links amino acids together to form proteins

-  Every strand of RNA is a sequence of **four** building blocks called **nucleotides**(A,U,G,C)

- Each RNA nucleotide consists of **three parts**: a sugar, a phosphate group, and a nitrogen-containing base(RNA shares **A,G,C** in common with DN)

- **Different bases form bonds of different strengths.** Guanine-Cytosine form the strongest bond, Adenine-Uracil form the next strongest bond, and Uracil-Guanine form the weakest bond.

### 1.2 Simple structure

- mRNAs do not usually fold into 3D shapes

### 1.3 Translating the mRNA code

- Every set of **three nucleotides** corresponds to a specific amino acid (amino acids are the building blocks of proteins)

- Every three nucleotides of mRNA are recognized by a particular transfer RNA (tRNA)

### 1.4 Types of RNAs and their functions

![3png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/16-22-26-16-3.png)

#### 1.4.1 Coding-RNA (messenger RNA; mRNA)

- determines the amino acid sequence in the protein produced.

- There are approximately 23,000 mRNAs encoded in human genome.

#### 1.4.2 Non-coding RNA (ncRNA)

Do not undergo translation to synthesize proteins. May hold the key to broaden our understanding of gene regulation and human diseases

- Ribosomal RNA (rRNA)

- Transfer RNA (tRNA; 80 nt)

- Small nuclear RNAs (snRNA; 150 nt)
  
  - Their primary function is to process the precursor mRNA (pre-mRNA).

- Small nucleolar RNAs (snoRNA; 60-300 nt)
  
  - responsible for sequence-specific nucleotide modification.

- Piwi-interacting RNAs (piRNA; 24-30 nt)
  
  - maintaining genome stability in germline cells.
  
  - gametogenesis

- MicroRNAs (miRNA; 21-22 nt)
  
  -  the most widely studied class of ncRNAs.

- Long noncoding RNAs (lncRNA)

### 1.5 The main RNA types in RRIs

![1.png](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-19-56-1.png)

### 1.6 Significance of RRIs

the significance of RRIs for **RNA function and regulation**

- It is studied that the **genome is largely transcribed**, and most transcripts do not encode proteins.

- Throughout their life cycle, RNAs function by interacting with different molecules (e.g., proteins, DNA, RNA).
  
  - **RBPs-RNAs**: perform specific functions in RNA processing and post-transcriptional regulation.
  
  - **lncRNAs-chromatin**:
    
    - regulate the expression of specific genes
    
    - modulate the epigenetic states of neighboring genomic loci.
  
  - **RNA-RNA**:
    
    -  In **splicing**, small nuclear RNAs (snRNAs) can recognize intronic regions of precursor messenger RNAs (premRNAs)
    
    - In **translation**, amino-acylated transfer RNAs (tRNAs) interact with mRNAs by reading the three-letter code and define protein amino acid sequences.
    
    - In **microRNA (miRNA) targeting**, the base pairing between a miRNA and 3′ UTR of an mRNA can lead to the degradation or translation inhibition of the mRNA.
    
    - In **RNA modification**, small nucleolar RNAs (snoRNAs) guide the modification of ribosomal RNAs (rRNAs) by base pairing.

- They can also be used to study all interactions concerning a certain RNA, usually combined with RNA pull-down with a set of target-specific antisense probes.

- RRIs can be used to represent **interactions between two RNAs** (inter-molecular) or **between different regions of one RNA molecule** (intra-molecular).
  
  - Intramolecular RRIs is the basis of forming RNA secondary structures.

> [2] [Predict_RNA-RNA_Interaction by Deep Learning](https://github.com/mohan-mj/Predict_RNA-RNA_Interaction)

## II Computational problems for RNA-RNA interaction

### 2.1 Binding sites prediction

#### Problem:

1. Intermolecular structure competes with the formation of self-structure in a concentration-dependent manner.

2. Predict only specific types of RNA–RNA interactions.

#### Step:

1. **Input**: (sequence & structure information)
   
   * one RNA
   
   * two RNAs: RNA1 & RNA2 (Target RNA)

2. **Pre-train**: Extract RNA features

3. **Model**: deep learning model & alignment based model

4. **Output**: (binary prediction)
   
   * **Site-level-based**: predicting the specific binding sites on RNAs
   
   * **RNA-level-based**: judges whether the input RNA pairs interact or not

#### Accuracy measures

<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-12-19-19-1.jpg" alt="1.jpg" width="242" data-align="center">

* **True positived(TPs):** the number of nucleotides on a correctly *predicted binding region*

* **False positive (FPs):** the number of nucleotides in a falsely *predicted binding region*

* **False negatives (FNs):** the number of nucleotides in a *binding region* where interactions are not predicted

* **True negatives (TNs):** not used (the number of true negatives grows exponentially
  with sequence length while TP, FP and FN grow linearly)

* **Metrics**: `accuracy, sensitivity, specificity, F-score, PPV(precision), NPV`
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-18-16-52-14.jpg" alt="14.jpg" width="317" data-align="center">

* **PPV**: more useful than the other metrics for evaluating binary classifiers on
  **imbalanced datasets** (high PPV means more robust)

* **sensitivity**: in the search for potential targets of specific miRNAs 

* **specificity**: in the examination of miRNAs that regulate specific genes

### 2.2 Binding preference prediction

/

## III Method for RRIs

<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-27-40-5.png" alt="5.png" data-align="center" width="499">

### 3.1 Traditional

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-36-08-1.jpg" title="" alt="1.jpg" data-align="center">

### 3.2 Deep Learning

#### 3.2.1 Timeline

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-22-20-11.jpg" title="" alt="11.jpg" data-align="center">

#### 3.2.2 Classfication of Methods

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-24-19-22.jpg" title="" alt="22.jpg" data-align="center">

#### 3.2.3 Deep learning Method for RRI

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-25-41-3.png" title="" alt="3.png" data-align="center">

## IV Benchmark Dataset

### 4.1 Sequence

| Name            | Last Update | Type(RNA-Target RNA) | URL                                                                                                                                                               | Method                                                                           |
|:---------------:|:-----------:|:--------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| ENCORI          | 2014        | RNA-RNA              | https://starbase.sysu.edu.cn/                                                                                                                                     |                                                                                  |
| RISE            | 2018        | RNA-RNA              | http://rise.zhanglab.net/                                                                                                                                         |                                                                                  |
| RNAInter        | 2020        | RNA-RNA              | [RNA interactome repository with increased coverage and annotation](http://www.rnainter.org/)                                                                     |                                                                                  |
| miRecords       | 2013        | miRNA-mRNA           | http://c1.accurascience.com/miRecords/                                                                                                                            | /                                                                                |
| mirMark         | 2014        | miRNA-mRNA           | [mirMark: a site-level and UTR-level classifier for miRNA target prediction](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0500-5)          | [DeepTarget](https://github.com/ailab-seoultech/deepTarget)                      |
| CLASH           | 2013        | miRNA-mRNA           | [Mapping the Human miRNA Interactome by CLASH Reveals Frequent Noncanonical Binding](https://www.sciencedirect.com/science/article/pii/S009286741300439X)         | [DeepMirTar](https://academic.oup.com/bioinformatics/article/34/22/3781/5026656) |
| PAR-CLIP        | 2010        | miRNA-mRNA           | /                                                                                                                                                                 | /                                                                                |
| lncRNASNP2      | 2017        | miRNA-lncRNA         | [lncRNASNP2: An updated database of functional SNPs and mutations in human and mouse lncRNAs. Nucleic Acids Res](http://bioinfo.life.hust.edu.cn/lncRNASNP#!/snp) |                                                                                  |
| miRTarBase      | 2018        | miRNA-lncRNA         | [miRTarBase update 2018: A resource for experimentally validated microRNA-target interactions](http://miRTarBase.mbc.nctu.edu.tw/)                                |                                                                                  |
| LncRNA2Target   | 2018        | lncRNA-mRNA/gene     | [A
comprehensive database for target genes of lncRNAs in human and mouse.](http://123.59.132.21/lncrna2target)                                                    |                                                                                  |
| C/D box snoRNAs | 2007        | sonRNA-rRNA          | [C/D box snoRNAs](https://people.biochem.umass.edu/fournierlab/snornadb/mastertable.php)                                                                          |                                                                                  |

### 4.2 RNA bimolecular Structure

<img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/17-22-33-14-11.png" title="" alt="11.png" data-align="center">

| Name          | Last Update | Tpye                  | URL                                                                                           |
|:-------------:|:-----------:|:---------------------:|:---------------------------------------------------------------------------------------------:|
| RNA Structure | 2012        | Bimolecular Structure | [RNA Structure](http://rna.urmc.rochester.edu/RNAstructureWeb/Servers/Predict2/Predict2.html) |

> [3] [Recent Deep Learning Methodology Development for RNA–RNA Interaction Prediction]((https://www.mdpi.com/2073-8994/14/7/1302))

## V Papers

### 5.1 miRNA target prediction

#### 5.1.1 DeepTarhet: miRNA-mRNA

* **What is the significance of miRNA Target prediction?**
  
  * miRNAs control the expression of target mRNAs, miRNA-mRNA pairs is
    of utmost importance in deciphering gene regulation
  
  * The **3’UTR of mRNA** is the region that directly follows the **translation termination codon**. By binding to target sites within the 3’ UTR, miRNAs can **decrease gene expression of mRNAs** by either **inhibiting translation** or **causing degradation of the transcript**

* **miRNA-mRNA interaction**
  
  <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-20-12-33-10.jpg" title="" alt="10.jpg" data-align="center">
  
  * **seed sequence**(miRNA): the first two to eight nucleotides starting from the 5’ to the 3’UTR
  
  * **candidate target site**: the small segment of length k of mRNA that contains a complementary match to the seed region at the head
  
  * **bind site**:
    
    * **canonical sites**: complementary to the miRNA seed region (exact W–C
      pairing of 2–7 or 3–8 nts of the miRNA)
    
    * **non-canonical sites**: pairing at positions 2–7 or 3–8, allowing G-U pairs and up to one bulged or mis-matched nucleotide

* **Dataset**
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-19-33-46-6.jpg" alt="6.jpg" width="404" data-align="center">
  
  1. soure: [mirMark: a site-level and UTR-level classifier for miRNA target prediction](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0500-5)
  
  2. Train: 2042 human miRNA-mRNA
     
     * Positive set:
       
       * site-level: 507
       
       * gene-level: 2,891
     
     * Negative set: (generated)
       
       - site-level: 507
       
       - gene-level: 3,313
       
       ---
       
       **HOW :**
       
       1. generated mock miRNAs
       
       2. mock miRNA-target pairs were then generated for each real miRNA-target pair in the positive dataset by replacing the positive miRNA in the real miRNA-target pair
       
       **WHY:**
       
       1. handle lack of biologically meaningful and experimentally verified negative pairs for training
       
       2. obtain a balanced training dataset
       
       ---
       
       

* **Model**: Modeling RNAs using RNN based Auto-encoder
  
  ![15.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-19-06-27-15.jpg)
  
  **Input:** miRNA and CTS sequence pair
  
  * **Layer:** 
    
    * layer 1: two **auto-encoders** for modeling miRNAs and mRNAs (learning RNAs representations of inherent features)
    
    * combine two representations into one: concatenating each dimension
      
      * Better method: learn **joint representations** that are shared across multiple modalities after learning **modality-specific network layers**
    
    * layer 2: **RNN** layer to model the interaction between miRNA and mRNA sequences (learn interaction features)
      
      ![9.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-20-08-53-9.jpg)
    
    * **Output:** classifying targets and non-targets (the probability of the given pair being a valid miRNA-target pair)

* **Results**
  
  ![8.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-19-59-27-8.jpg)

* **Future improve**
  
  * RNAs Representstion part
  
  * Combine two sequence(learn joint representations)
  
  * Interaction learning part

#### 5.1.2 DeepMirTar: miRNA-mRNA (site-level)

* **Contributions:**
  
  * Add **prior/expert knowledge**: features including high level expert-designed, low-level expert-designed and raw-data-level
    
    <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-21-21-09-16.jpg" alt="16.jpg" width="352" data-align="center">

* **Dataset**: 
  
  * **mirMark + CLASH**
    
    * **mirMark** : 507 miRNA-target-site pairs
    
    * **CLASH** :  18,514 miRNA-target-site pairs
  
  * **Train**:
    
    * **Positive**: 3915 duplexes
      
      * mirMark: 473 duplexe
      
      * CLASH: 3442 duplexes
    
    * **Negative**

* **Model**
  
  ![19.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/19-00-33-59-19.jpg)

* **Results**
  
  ![20.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/19-00-52-59-20.jpg)
  
  

### 5.2 miRNA target prediction

#### 5.2.1 DeepLGP: lncRNA-genes

* **Contributions**:
  
  * **Meaningful Research**: genes targeted by the same lncRNA sets had a strong likelihood of causing the same diseases, which could help in **identifying disease-causing PCGs**.
  
  * **Feature extraction**: 
    
    * Feature Selection: gene and lncRNA features
    
    * GCN: interaction features (Network encoding)

* **Model**:
  
  ![18.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/19-00-26-20-18.jpg)

* **Result**
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/19-00-56-51-21.jpg" alt="21.jpg" width="311" data-align="center">

#### 5.2.2 Predicting the interaction biomolecule types for lncRNA: an ensemble deep learning approach

#### 5.2.3 Predicting microRNA–disease associations from lncRNA–microRNA interactions via Multiview Multitask Learning

# Appendix

## I Dataset(RRI)

### 1.1 ENCORI

<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-49-33-ENCORI_workflow.png" alt="ENCORI_workflow.png" width="523" data-align="center">

<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-14-49-51-11.jpg" alt="11.jpg" width="334" data-align="center">

### 1.2 RISE

- RRIs in RISE come from **5 species** --human, mouse, yeast, *E.coli, S.enteria,* or **10** different **cell lines**.

- RISE curated RRIs from **3 types** of sources:
  
  - **transcriptome-wide studies**
    
    - use PARIS, MARIOS, SPLASH, LIGR-seq technologies to discover RRIs
  
  - **targeted studies**
    
    - use RIA-seq, RAP-RNA, CLASH
  
  - **other databases/datasets**
    
    -  include RRIs from the NPInter v3.0, RAID v2.0, RAIN databases, and the *Lai D, et al.* 2016 datasets

![2jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/17-22-52-54-2.jpg)

![1jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/17-23-18-47-1.jpg)

### 1.3 RNAIter

![homejpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-15-16-41-home.jpg)

![13jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-15-22-28-13.jpg)

### 1.4 lncRNASNP2

![12.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/18-15-03-23-12.jpg)

### 1.5 C/D box snoRNAs

![1.jpg](https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/08/17-23-52-23-1.jpg)
