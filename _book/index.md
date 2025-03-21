---
title: "Data Science Foundations and Machine Learning Using R"
author: "[<span style='color:white'>Reza Mohammadi</span>](https://www.uva.nl/profile/a.mohammadi)"
date: "March 12, 2025"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
link-citations: yes
fontsize: 11pt
geometry: a4paper
colorlinks: yes
lot: false
lof: false
always_allow_html: yes
twitter-handle: LiveR
graphics: yes
url: https://uncovering-data-science.netlify.app
github-repo: "RezaMoammadi/Book-Data-Science"
cover-image: "images/logo_black.png"
description: "An intuitive and practical introduction to the exciting field of Data Science and Machine Learning"
editor_options: 
  markdown: 
    wrap: sentence
---

# Welcome {-}  

Welcome to the online companion for *Data Science Foundations and Machine Learning Using R*, a hands-on and intuitive introduction to the exciting field of data science—no prior programming or analytics experience required.  

Inspired by the principles of open science and reproducibility, this book embraces transparency. The code, datasets, and materials are freely available on [CRAN](https://cran.r-project.org/web/packages/liver/index.html) through the **liver** package (<https://CRAN.R-project.org/package=liver>), enabling readers to engage with the content interactively.  

<a href="https://CRAN.R-project.org/package=liver"><img src="images/logo.png" alt="Book cover" width="250" height="250" align="right" style="margin: 0 1em 0 1em"/></a>

The latest version of this book is available online at [https://uncovering-data-science.netlify.app](https://uncovering-data-science.netlify.app). The source code for this site is maintained on [GitHub](https://github.com/RezaMoammadi/Book-Data-Science), ensuring continuous updates and improvements.

### Work in Progress {-}  

This book is an evolving project, and we welcome feedback from readers. If you have suggestions, corrections, or ideas for improvement, we encourage you to reach out:  

- **Email**: [Contact Us](mailto:a.mohammadi@uva.nl)  
- **Contribute on GitHub**: Submit an issue or pull request at [GitHub Repository](https://github.com/RezaMoammadi/Book-Data-Science)  

Your input is invaluable in making this book a better resource for the data science community!  

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/RezaMoammadi/Book-Data-Science">Book</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.uva.nl/profile/a.mohammadi">Reza Mohammadi</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
                  
<hr>

<a href="https://www.netlify.com"><img src="https://www.netlify.com/img/global/badges/netlify-color-accent.svg"/></a>
The book website is hosted on [Netlify](https://www.netlify.com).

# Preface {.unnumbered}  

Data science is transforming the way we solve problems, make decisions, and uncover insights from data. From generative AI chatbots—such as ChatGPT, DeepSeek, and Gemini—to personalized recommendations on Netflix and fraud detection in banking, data-driven techniques are revolutionizing industries. Whether you are new to data science or an experienced professional, *Data Science Foundations and Machine Learning Using R* provides an accessible, hands-on introduction to this exciting field—no prior analytics or programming experience required.  

This book is designed for those who want to understand data science and machine learning, as well as develop data-driven solutions for real-world applications. It provides a structured and accessible introduction to the fundamental concepts, making it suitable for beginners while also equipping readers with the skills to apply machine learning algorithms effectively. Whether you are a student, a business professional, or a researcher, this book offers a practical and intuitive approach to learning data science using **R**. Having taught these topics myself, I have found this hands-on approach to be highly effective and well-received by students.  

Unlike many theoretical texts, this book emphasizes a *learning-by-doing* approach. Each topic is introduced through step-by-step explanations, practical examples, and exercises—some worked out manually to build intuition, while others involve coding implementations in **R**. Readers will not only develop a solid theoretical foundation in machine learning and intelligent systems but also gain experience with widely used tools and techniques for applying these concepts in practice.  

The remainder of this preface outlines the book’s structure, how it can be used in different learning environments, and the foundational knowledge needed to get the most out of it.  

## Why This Book? {.unnumbered}  

Data science is a rapidly evolving field that integrates machine learning, statistical techniques, and computational tools to extract meaningful insights from data. This book provides a structured introduction to data analysis and machine learning using **R**, a powerful open-source programming language widely adopted for statistical computing, visualization, and predictive modeling.  

Unlike many textbooks that assume prior programming experience, this book is designed to be **intuitive and hands-on**, engaging readers through real-world applications to accelerate the learning curve. Concepts are introduced step by step and immediately applied to real datasets, reinforcing both theoretical foundations and practical skills. Readers will not only learn key data science techniques but also gain hands-on experience in solving analytical problems.  

Compared to proprietary software like SAS or SPSS, **R** offers a free, flexible, and highly extensible environment for data science. With its vast ecosystem of packages, **R** is a preferred choice for data mining, machine learning, and exploratory analysis across academia, industry, and research.  

## Who Should Read This Book? {.unnumbered}  

This book is designed for anyone interested in learning data science and machine learning, particularly those new to the field. It is suitable for:  

- Business professionals seeking to leverage data for better decision-making,  
- Students and researchers applying data analysis in their studies or projects,  
- Beginners with no prior experience in programming or analytics,  
- Anyone interested in exploring data science and machine learning using **R**.  

Whether used for self-study, in a classroom setting, or as a reference for professionals, this book provides a practical and structured approach to applying data science techniques in real-world scenarios.  

## What You Will Learn {.unnumbered}  

This book provides a structured, hands-on introduction to data science and machine learning using **R**. Through practical examples and real-world datasets, you will:  

- *Gain proficiency in R programming* – Master the core syntax, data structures, and essential functions for data analysis.  
- *Implement the data science workflow* – Apply a structured approach to data cleaning, transformation, and exploration.  
- *Uncover patterns through exploratory data analysis (EDA)* – Use statistical and graphical techniques to identify insights.  
- *Build predictive models* – Develop machine learning models for classification, regression, and clustering.  
- *Assess and refine models* – Evaluate performance using key metrics and optimize algorithms for better accuracy.  
- *Apply data science techniques in real-world scenarios* – Work with datasets from marketing, finance, and other domains to solve practical problems.  

By the end of this book, you will have a strong foundation in data science and machine learning, equipped with the skills to analyze complex datasets and develop effective data-driven solutions.  

## How This Book Is Structured {.unnumbered}  

This book is designed as a *hands-on guide*, progressing from foundational concepts to more advanced machine learning techniques. Each chapter builds upon previous topics, ensuring a logical learning experience and reinforcing practical skills.  

To illustrate concepts effectively, we use *real-world datasets* (see Table \@ref(tab:data-table)), which are available in the **liver** package. These datasets make it easy for readers to follow along with examples and exercises. Below is an overview of the book’s chapters:  

- *Chapter \@ref(chapter-into-R)* – Introduction to **R**, including installation and essential functions.  
- *Chapter \@ref(chapter-intro-DS)* – The foundations of data science and its methodology.  
- *Chapter \@ref(chapter-data-prep)* – Techniques for cleaning and transforming data.  
- *Chapter \@ref(chapter-EDA)* – Exploratory Data Analysis (EDA) using visualization and summary statistics.  
- *Chapter \@ref(chapter-statistics)* – Fundamentals of statistical analysis and hypothesis testing.  
- *Chapter \@ref(chapter-modeling)* – Overview of machine learning models.  
- *Chapter \@ref(chapter-knn)* – The k-Nearest Neighbors (k-NN) algorithm.  
- *Chapter \@ref(chapter-evaluation)* – Model evaluation techniques and performance metrics.  
- *Chapter \@ref(chapter-bayes)* – The Naïve Bayes classifier for probabilistic modeling.  
- *Chapter \@ref(chapter-regression)* – Linear regression for predictive modeling.  
- *Chapter \@ref(chapter-tree)* – Decision trees and ensemble methods like Random Forests.  
- *Chapter \@ref(chapter-nn)* – Introduction to neural networks.  
- *Chapter \@ref(chapter-cluster)* – Clustering techniques such as k-means.  

Each chapter concludes with *practical exercises* that reinforce key concepts and provide hands-on experience with real-world datasets. These exercises are designed to help readers apply what they’ve learned, deepening their understanding of data science and machine learning.

## How to Use This Book {.unnumbered}  

This book is designed for *self-study, classroom instruction, and professional learning*. Readers can follow the chapters sequentially for a structured learning experience or refer to specific topics as needed.  

To maximize learning and practical engagement:  

- *Run the Code Examples* – All examples are designed for interactive execution in **R** to reinforce key concepts.  
- *Complete the Exercises* – Practical exercises at the end of each chapter strengthen understanding and problem-solving skills.  
- *Modify and Experiment* – Tweaking code and testing variations enhance comprehension and adaptability.  
- *Use It as a Reference* – Once familiar with the fundamentals, this book serves as a valuable resource for real-world applications.  

This book has been successfully used in *data science courses at the University of Amsterdam* and is well-suited for similar academic programs and professional training initiatives.  

## Datasets Used in This Book {.unnumbered}  

This book incorporates real-world datasets to illustrate key data science and machine learning concepts. Table \@ref(tab:data-table) provides an overview of the datasets used throughout the book, all of which are included in the **liver** package.  

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
   <td style="text-align:left;"> Chapters 7, 12 </td>
  </tr>
  <tr>
   <td style="text-align:left;font-weight: bold;color: black !important;"> adult </td>
   <td style="text-align:left;width: 20em; "> US Census data for income prediction. </td>
   <td style="text-align:left;"> Chapters 3, 11 </td>
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



These datasets were selected to provide hands-on exposure to real-world problems across domains such as marketing, finance, and predictive modeling. By working with these datasets, readers will develop the practical skills necessary to apply data science techniques effectively in professional settings.

The datasets listed in Table \@ref(tab:data-table) are primarily used in case studies throughout the book. Additionally, the **liver** package contains over 15 datasets, providing readers with further opportunities for exploration and hands-on practice. While the main text focuses on a subset of these datasets, additional datasets are integrated into the exercises at the end of each chapter, allowing readers to extend their learning beyond the core examples presented in the book.  

## Using This Book for Teaching {.unnumbered}  

This book is well-suited for *introductory data science and machine learning courses* or as a *supplementary resource* for analytics training. Its structured approach, hands-on exercises, and real-world case studies make it an effective learning tool for students at various levels.  

To support structured learning, the book includes *over 500 exercises*, categorized into three levels:  

- *Conceptual Exercises* – Reinforce theoretical understanding and fundamental principles.  
- *Applied Exercises* – Require hands-on data analysis using real-world datasets.  
- *Advanced Exercises* – Explore complex applications and machine learning techniques, encouraging deeper engagement with the material.  

Instructors using this book will have access to supplementary teaching materials, including an instructor’s manual, slides, and test banks. These resources provide a comprehensive framework for delivering data science and machine learning courses, whether in academic settings or professional training programs.  
