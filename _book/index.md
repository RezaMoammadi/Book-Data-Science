---
title: "Uncovering Data Science with R"
author: "Reza Mohammadi"
date: "2025-02-19"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
url: https://uncovering-data-science.netlify.app
cover-image: "images/cover.jpg"
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
editor_options: 
  markdown: 
    wrap: sentence
---

# Preface {.unnumbered}  

**Data science is transforming the way we solve problems, make decisions, and uncover insights from data.** Whether you're a beginner or an experienced professional, *Uncovering Data Science with R* provides an intuitive and practical introduction to this exciting field—no prior analytics or programming experience required.  

This book is a work in progress, and we welcome feedback from readers. If you have any comments, suggestions, or corrections, please feel free to contact us at [Contact Us](mailto:a.mohammadi@uva.nl).  

## Why This Book? {.unnumbered}  

Data science is a rapidly evolving field that leverages computational tools and techniques to transform raw data into actionable insights. In this book, we introduce the fundamental skills needed to work with **R**, a powerful and freely available statistical programming language widely used for data analysis, visualization, and machine learning.  

Unlike many other books on data science, our focus is on accessibility. We aim to provide an intuitive and practical introduction, making **R** and data science concepts understandable for those with little or no technical background. Our hands-on approach ensures that you will not only learn theoretical concepts but also gain experience applying them to real-world datasets.  

Compared to commercial software like SAS or SPSS, **R** provides a free, open-source, and highly extensible platform for statistical computing and machine learning. Its rich ecosystem of packages makes it an excellent alternative to proprietary data mining tools.  

Inspired by the Free and Open Source Software (FOSS) movement, the content of this book is open and transparent, ensuring reproducibility. The code, datasets, and materials are hosted on [CRAN](https://cran.r-project.org/web/packages/liver/index.html) and are accessible via the **liver** package (<https://CRAN.R-project.org/package=liver>), allowing readers to engage with the book interactively.  

## Who Should Read This Book? {.unnumbered}  

This book is for anyone interested in learning about data science, particularly those new to the field. It is designed for:  

- Business professionals who want to leverage data for decision-making,  
- Students and researchers looking to apply data analysis in their work,  
- Beginners with no prior programming experience,  
- Anyone interested in data science and machine learning using **R**.  

## What You Will Learn {.unnumbered}  

The primary goal of this book is to introduce data science concepts using **R** as a tool for data analysis and machine learning. **R** is an open-source language and environment for statistical computing and graphics, offering a vast collection of packages for data mining, visualization, and modeling.  

Through hands-on examples and real-world datasets, you will learn:  

- The basics of **R** and how to set up your environment,  
- The core principles of data science and the Data Science Methodology,  
- How to clean, transform, and explore data,  
- The fundamentals of statistical analysis, machine learning, and data visualization,  
- How to build and evaluate machine learning models, including classification, regression, clustering, and neural networks,  
- How to apply these techniques to real-world datasets.  

## The Data Science Process {.unnumbered}  

Data science follows an iterative and structured methodology for analyzing and extracting insights from data. This book follows this framework:  

1. **Problem Understanding** – Defining the objective and understanding the data.  
2. **Data Preparation** – Preparing raw data for analysis.  
3. **Exploratory Data Analysis (EDA)** – Identifying patterns and relationships in data.  
4. **Preparing Data for Modeling** – Transforming data for machine learning models.  
5. **Modeling** – Building predictive models using machine learning algorithms.  
6. **Evaluation** – Assessing model performance using various metrics.  
7. **Deployment** – Applying the trained model to real-world scenarios.  

By the end of this book, you will have a solid understanding of these phases and be able to apply them effectively.  

## How This Book Is Structured {.unnumbered}  

This book is structured as a **hands-on guide**, designed to take you from beginner to practitioner in **R** and data science. The chapters follow a logical progression, starting with foundational concepts and gradually introducing more advanced techniques.  

We use **real-world datasets** (see Table \@ref(tab:data-table)) throughout the book to illustrate key concepts. These datasets are available in the **liver** package and can be accessed easily. Below is a brief overview of the book’s chapters:  

- **Chapter \@ref(chapter-into-R)** – Introduction to **R**, including installation and basic operations.  
- **Chapter \@ref(chapter-intro-DS)** – Introduction to Data Science and its methodology.  
- **Chapter \@ref(chapter-data-prep)** – Data preparation techniques.  
- **Chapter \@ref(chapter-EDA)** – Exploratory Data Analysis (EDA) using visualization and summary statistics.  
- **Chapter \@ref(chapter-statistics)** – Basics of statistical analysis, including descriptive statistics and hypothesis testing.  
- **Chapter \@ref(chapter-modeling)** – Overview of machine learning models.  
- **Chapter \@ref(chapter-knn)** – k-Nearest Neighbors (k-NN) algorithm.  
- **Chapter \@ref(chapter-evaluation)** – Model evaluation metrics and techniques.  
- **Chapter \@ref(chapter-bayes)** – Naïve Bayes classifier for probabilistic modeling.  
- **Chapter \@ref(chapter-regression)** – Linear regression for predictive modeling.  
- **Chapter \@ref(chapter-tree)** – Decision trees and Random Forests.  
- **Chapter \@ref(chapter-nn)** – Neural networks and deep learning basics.  
- **Chapter \@ref(chapter-cluster)** – Clustering techniques, including k-means.  

At the end of each chapter, you will find **practical exercises and labs** to reinforce your learning. These exercises use real-world datasets and provide step-by-step guidance to ensure hands-on experience.  

## How to Use This Book {.unnumbered}  

This book is designed for **both self-study and classroom use**. You can read it cover to cover or jump to the chapters that interest you most. Each chapter builds on the previous ones, so beginners are encouraged to follow the sequence for a smooth learning experience.  

To get the most out of this book:  

- **Run the code examples** – All code snippets are designed to be executed interactively in **R**.  
- **Complete the exercises** – Practical exercises reinforce key concepts and improve problem-solving skills.  
- **Modify and experiment** – Try changing the code to explore different scenarios.  
- **Use it as a reference** – Once you're familiar with the basics, use this book as a guide for working with real-world data.  

This book has also been used in **data science courses** at the University of Amsterdam. It can serve as a textbook for similar courses or as a supplementary resource for more advanced analytics training.  

## Datasets Used in This Book {.unnumbered}  

Table \@ref(tab:data-table) lists the datasets used in this book. These real-world datasets are used to illustrate key concepts and are available in the **liver** package, which can be downloaded from CRAN.  

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<caption>(\#tab:data-table)List of datasets used for case studies in different chapters. Available in the R package liver.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Name </th>
   <th style="text-align:left;"> Description </th>
   <th style="text-align:left;"> Chapter </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> churn </td>
   <td style="text-align:left;width: 20em; "> Customer churn dataset. </td>
   <td style="text-align:left;"> Chapters 4, 6, 7, 8, 10 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> bank </td>
   <td style="text-align:left;width: 20em; "> Direct marketing data from a Portuguese bank. </td>
   <td style="text-align:left;"> Chapter 7, 12 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> adult </td>
   <td style="text-align:left;width: 20em; "> US Census data for income prediction. </td>
   <td style="text-align:left;"> Chapter 11 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> risk </td>
   <td style="text-align:left;width: 20em; "> Credit risk dataset. </td>
   <td style="text-align:left;"> Chapter 9 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> marketing </td>
   <td style="text-align:left;width: 20em; "> Marketing campaign performance data. </td>
   <td style="text-align:left;"> Chapter 10 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> house </td>
   <td style="text-align:left;width: 20em; "> House price prediction dataset. </td>
   <td style="text-align:left;"> Chapter 10 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> diamonds </td>
   <td style="text-align:left;width: 20em; "> Diamond pricing dataset. </td>
   <td style="text-align:left;"> Chapter 3 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> cereal </td>
   <td style="text-align:left;width: 20em; "> Nutritional information for 77 breakfast cereals. </td>
   <td style="text-align:left;"> Chapter 13 </td>
  </tr>
</tbody>
</table>



## Prerequisites {.unnumbered}  

No prior programming experience is required, but a basic understanding of numbers and logic will be helpful. To run the code in this book, you need to install **R**, **RStudio**, and several **R** packages.  






