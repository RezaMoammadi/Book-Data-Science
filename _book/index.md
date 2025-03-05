---
title: "Data Science Foundations and Machine Learning Using R"
author: "[<span style='color:white'>Reza Mohammadi</span>](https://www.uva.nl/profile/a.mohammadi)"
date: "2025-03-05"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
url: https://uncovering-data-science.netlify.app
github-repo: "RezaMoammadi/Book-Data-Science"
cover-image: "images/logo_black.png"
description: "An intuitive and practical introduction to the exciting field of Data Science and Machine Learning"
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
editor_options: 
  markdown: 
    wrap: sentence
---



# Preface {.unnumbered}  

Data science is transforming the way we solve problems, make decisions, and uncover insights from data. From personalized recommendations on Netflix to fraud detection in banking, data-driven techniques are revolutionizing industries. Whether you are new to data science or an experienced professional, *Data Science Foundations and Machine Learning Using R* provides an accessible, hands-on introduction to this exciting field—no prior analytics or programming experience required.  

## Why this book? {.unnumbered}  

Data science is an evolving discipline that integrates computational tools and statistical techniques to transform raw data into actionable insights. This book introduces the fundamental skills needed to analyze data and build machine learning models using **R**, a powerful open-source statistical programming language widely used for data science, machine learning, and visualization.  

Unlike many textbooks that assume prior programming knowledge, this book is designed to be accessible. The focus is on **intuitive learning**, ensuring that concepts are not just explained but applied through practical, real-world examples. Readers will gain hands-on experience working with real datasets, developing essential data science skills while reinforcing theoretical foundations.  

Compared to proprietary software like SAS or SPSS, **R** offers a free, highly extensible ecosystem for statistical computing and machine learning. Its rich collection of packages makes it a widely adopted tool for data mining, predictive modeling, and exploratory data analysis.  

## Who should read this book? {.unnumbered}  

This book is intended for:  

- Business professionals looking to leverage data for decision-making,  
- Students and researchers applying data analysis in their work,  
- Beginners with no prior experience in programming or analytics,  
- Anyone interested in data science and machine learning using **R**.  

It is suitable for self-study, classroom use, or as a reference for professionals seeking to apply data science techniques in practical settings.  

## What you will learn {.unnumbered}  

This book introduces data science principles through a structured, hands-on approach using **R**. Topics covered include:  

- The fundamentals of **R** programming,  
- The data science workflow, including data wrangling, transformation, and visualization,  
- Exploratory data analysis (EDA) and statistical inference,  
- Machine learning techniques, including classification, regression, clustering, and neural networks,  
- Model evaluation and performance metrics,  
- Practical applications using real-world datasets.  

## The data science workflow {.unnumbered}  

Data science follows a structured methodology for analyzing and extracting insights from data. This book adheres to this process:  

1. **Problem Understanding** – Defining the objective and understanding the data.  
2. **Data Preparation** – Cleaning and transforming raw data for analysis.  
3. **Exploratory Data Analysis (EDA)** – Identifying patterns and relationships within the data.  
4. **Preparing Data for Modeling** – Preparing data for predictive modeling.  
5. **Modeling** – Building and training machine learning models.  
6. **Evaluation** – Assessing model performance using various metrics.  
7. **Deployment** – Applying trained models to real-world scenarios.  

By the end of this book, you will be familiar with each phase of this process and be able to apply it effectively in practice.  
## How this book is structured {.unnumbered}  

The book is structured as a **hands-on guide**, progressing from foundational concepts to more advanced machine learning techniques. Each chapter builds upon previous topics, ensuring a logical learning experience.  

We use **real-world datasets** (see Table \@ref(tab:data-table)) to demonstrate key concepts. These datasets are available in the **liver** package, making it easy to follow along. Below is an overview of the book’s chapters:  

- **Chapter \@ref(chapter-into-R)** – Introduction to **R** and its essential functions.  
- **Chapter \@ref(chapter-intro-DS)** – The foundations of data science and its methodology.  
- **Chapter \@ref(chapter-data-prep)** – Techniques for cleaning and transforming data.  
- **Chapter \@ref(chapter-EDA)** – Exploratory Data Analysis (EDA) using visualization and summary statistics.  
- **Chapter \@ref(chapter-statistics)** – Fundamentals of statistical analysis and hypothesis testing.  
- **Chapter \@ref(chapter-modeling)** – Overview of machine learning models.  
- **Chapter \@ref(chapter-knn)** – k-Nearest Neighbors (k-NN) algorithm.  
- **Chapter \@ref(chapter-evaluation)** – Model evaluation techniques.  
- **Chapter \@ref(chapter-bayes)** – The Naïve Bayes classifier.  
- **Chapter \@ref(chapter-regression)** – Linear regression for predictive modeling.  
- **Chapter \@ref(chapter-tree)** – Decision trees and ensemble methods like Random Forests.  
- **Chapter \@ref(chapter-nn)** – Introduction to neural networks.  
- **Chapter \@ref(chapter-cluster)** – Clustering techniques such as k-means.  

At the end of each chapter, **practical exercises and coding labs** reinforce key concepts and provide hands-on experience with real-world datasets.  

## How to use this book {.unnumbered}  

This book is designed for **self-study, classroom instruction, or professional learning**. Readers can either follow the book sequentially or jump to specific topics as needed.  

To maximize learning:  

- **Run the code examples** – All examples are designed for interactive execution in **R**.  
- **Complete the exercises** – Practical exercises strengthen understanding and problem-solving skills.  
- **Modify and experiment** – Tweaking code allows deeper exploration of concepts.  
- **Use it as a reference** – Once familiar with the basics, refer to this book for real-world applications.  

The book has been used in **data science courses at the University of Amsterdam** and is suitable for similar courses in other institutions.  

## Datasets used in this book {.unnumbered}  

Table \@ref(tab:data-table) lists the datasets used throughout the book, all of which are included in the **liver** package.  

\begin{table}
\centering
\caption{(\#tab:data-table)List of datasets used for case studies in different chapters. Available in the R package liver.}
\centering
\begin{tabular}[t]{>{}l>{\raggedright\arraybackslash}p{20em}l}
\toprule
Name & Description & Chapter\\
\midrule
\textcolor{black}{\textbf{churn}} & Customer churn dataset. & Chapters 4, 6, 7, 8, 10\\
\textcolor{black}{\textbf{bank}} & Direct marketing data from a Portuguese bank. & Chapters 7, 12\\
\textcolor{black}{\textbf{adult}} & US Census data for income prediction. & Chapters 3, 11\\
\textcolor{black}{\textbf{risk}} & Credit risk dataset. & Chapter 9\\
\textcolor{black}{\textbf{marketing}} & Marketing campaign performance data. & Chapter 10\\
\addlinespace
\textcolor{black}{\textbf{house}} & House price prediction dataset. & Chapter 10\\
\textcolor{black}{\textbf{diamonds}} & Diamond pricing dataset. & Chapter 3\\
\textcolor{black}{\textbf{cereal}} & Nutritional information for 77 breakfast cereals. & Chapter 13\\
\bottomrule
\end{tabular}
\end{table}



## Using this book for teaching {.unnumbered}  

This book is suitable for **introductory data science and machine learning courses** or as a **supplementary resource** for analytics training. It includes **over 500 exercises** categorized into three levels:  

- **Conceptual Exercises** – Reinforce theoretical understanding.  
- **Applied Exercises** – Require hands-on analysis of real-world datasets.  
- **Advanced Exercises** – Explore complex applications and machine learning techniques.  

Faculty adopters have access to additional resources:  

- **Instructor’s Manual** – Solutions to all exercises, plus teaching guidance.  
- **PowerPoint Slides** – Presentation slides for each chapter.  
- **Test Bank** – Multiple-choice and short-answer questions.  
- **Data Sets** – Available in the **liver** package for easy access.  
