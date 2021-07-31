---
layout: post
title: How to read Machine Learning and Deep Learning Research papers
date: 2021-07-31 15:09:00
description: Tips on preparing Literature survey of a field and how to read a ML / DL research papers
comments: true
---

## Introduction

**How to read a research paper,** is probably the most important skill which any one who is into research or even anyone who wishes to be updated in the field with latest advancements has to master. When someone thinks of starting out in a domain, the first advice that comes is to look for relevant literature in the domain and read papers to develop an understanding of the domain. Papers are the most reliable and updated source of information about a particular domain. A research paper is a result of days of brainstorming of ideas, and structured and systematic experimentation to express an approach.

But why is reading papers considered such an important skill to be learnt ! Why is even reading papers necessary. Let's take on some motivation as to why is reading papers important to keep-up with the latest advances.
> This article is the summary of a talk that I delivered for the Introductory Paper Reading Session generously supported by [Weights and Biases](https://wandb.ai/site)  whose recorded version can be found here and slides can be found [here](https://docs.google.com/presentation/d/1wjNc3gdC21llAbS6w1YsF7UuUyXnPXQT6wmljBEtzV8/edit?usp=sharing).
## Dynamically Expanding field of Deep Learning
The field of deep learning has grown very rapidly in the recent years. We can quantify growth in a field by theh number of papers that come up everyday. Here is an illustration from one of the studies by ArXiv which is one of the platform where almost all of the papers, whether published or unpublished are putup.
<p align="center" width="100%">
	<a  href="https://blog.arxiv.org/2019/12/05/arxiv-machine-learning-classification-guide/"  title="Redirect to homepage">
		<img width="100%" src="https://raw.githubusercontent.com/saiamrit/technical-blog/master/images/read_papers1.png"> 
<figcaption align = "center">Fig.1 - Average. No. of machine learning papers uploaded to archive  every month</figcaption>
	</a>
</p>
From the figure we can see that the average no of papers has grown to 5X averaging from 300 papers per month in 2017 to around 1500 papers per month in 2019. The figure would probably be close to or above 2k papers per month in 2021. This is a huge number of papers coming up everyday. This shows how dynamic the field is at the current time and it is just growing exponentially in terms of number of papers and amount of new ideas and experiments coming up everyday.

Let's look at another figure from another study by arXiv
<p align="center" width="100%">
	<a  href="https://arxiv.org/help/stats/2018_by_area#cs_yearly"  title="Redirect to homepage">
		<img width="100%" src="https://raw.githubusercontent.com/saiamrit/technical-blog/master/images/read_papers2.png"> 
<figcaption align = "center">Fig.2 - Growth of Computer Science Papers on arXiv</figcaption>
	</a>
</p>
From the figure, the number of papers in the field of Computer Science has grown like a steep exponential curveand we see that around 36k papers come out each year out of which around 24k of them as we saw in the previous section are ij the field of ML and DL. We can also see in both the figures that the DL field in Green and CV in yellow are among the dominant areas in terms of percentages of papers coming out every year since the early 2000s while the field of CV has grown and opened up a lot after 2012 probably when the prominent work on [Image classification by deep networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) showed significant performance. These studies definitely speak how fast the field of computer Science is growing and amongst it, how the sub areas related to Machine Learning and Deep Learning are evolving too.

I hope these give a good idea of how fast the field has been evolving and would continue to evolve even faster in the future. But in this fast evolving field, **How can we keep up with the pace and develop a expertise in the field ?**

Quoting **[Dr. Jennifer Raff](https://anthropology.ku.edu/jennifer-raff)** 
>*" To form a truly educated opinion on a scientific subject, you need to become familiar with current research in that field. 
>And to be able to distinguish between good and bad interpretations of research, you have to be willing and able to read the primary research literature for yourself. "*

## Why to read research Papers

 - **To have a better grasp and understanding of the field:** For a particular field, there may be a lot of video lectures and books but with the rate at which the field has been growing, no book or video lecture can accomodate the latest information as soon as they get published. So research papers provide the most updated and reliable information in the field.
 
 - **To be able to contribute to the field in terms of novel ideas:** When we start working in a field, the first thing that we are advised to do is to do an extensive literature survey, going through all of the latest papers that have come up in the field till date. That is advised because we can have a very good understanding of the directions of works in the field and how the people actively working in the field are thinking. Only then we can start coming up with our own ideas to experiment upon.
 -  **To develop confidence in the field:** Once we start learning about the latest works in the field and we start to develop a good understanding by performing a extensive literature survey, we start developing more confidence to perform more experiments and exploring deeper in the field.
 -  **Most condensed and authentic source of latest knowledge in the field:** A reseach paper comes out of days and months, or some times even years of brainstorming of ideas, performing extensive experiments and validating the expected outcomes. The condensed experiments and thoughts is what is best expressed in a research paper that the authors write. Any new content that comes in the field in terms of state-of-the-art works is through research papers. Research papers are the source through which works that push the limits of knowledge in a field come up.

**Motivated enough ?**

Now that we have attained enough motivations as to why we should read research papers, lets look at how to do literature survey in a domain.  

**Let's do it !**
## Literature survey of a domain

The basic steps to perform literature survey in a field are the following:
 1.  Assemble collections of resources in the form of research papers, Medium articles, blog posts, videos, GitHub repository etc.
    
2.  Conduct a deep dive to classify the relevant and irrelevant material.
    
3.  Take structured notes that summarises the key discoveries, findings and techniques within a paper.

We shall take Pose Estimation as a example domain and understand each step.
### Step 1: Assembling all available resources 
First of all we collect all the resources in the form of blog posts, github repositories, medium articles and research papers available in the field, for our case it's pose estimation. The important question here is, **where can we find relevant resources in the field ?**

Following are sources where we can find the latest papers and resources:
-   [Twitter](https://twitter.com/AndrewYNg?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor): We can follow top researchers, groups and labs actively working and publishing in our field of domain and be updated with what they are currently working on.
    
-   [ML subreddit](https://www.reddit.com/r/MachineLearning/)
    
-   [arXiv](https://arxiv.org/): Platform where almost all of the papers be it accepted to a conference or not, are uploaded.
    
-   [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/): Created by [Anderj Karpathy](https://twitter.com/karpathy?lang=en) which used ML techniques to suggest relevant papers based on previous searches and interests.
    
-   [Papers With Code](https://paperswithcode.com/): Redirects to the paper's abstract page on arXiv, open source implementation of the papers along with links to datasets used and a lot of other analysis and meta information like the current state-of-the art method, comparision of performance of all previous methods in the field e.t.c.
    
-   Top ML, DL Conferences ([CVPR](https://openaccess.thecvf.com/menu), [ICCV](https://openaccess.thecvf.com/menu), [NeurIPS](https://nips.cc/), [ICML](https://icml.cc/), [ICLR](https://iclr.cc/) etc): Proceedings of the following conferences are a great place to look for latest accepted works in the domains accepted by the conference.
    
-   [Google Search](http://www.google.com)

Once listed down all the papers that we wish to look at and all resources we could find be it relevant or irrelevant, a table of this format shown in figure 3 can be prepared and in the first column, all the resources collected can be listed down.

<p align="center" width="100%">
	<a  href="https://towardsdatascience.com/how-you-should-read-research-papers-according-to-andrew-ng-stanford-deep-learning-lectures-98ecbd3ccfb3"  title="Redirect to homepage">
		<img width="100%" src="https://raw.githubusercontent.com/saiamrit/technical-blog/master/images/read_papers4.png"> 
<figcaption align = "center">Fig.3 - Initial Literature Survey chart for Pose Estimation</figcaption>
	</a>
</p>

### Step2 - Filtering out relevant and Irrelevant resources

Once listed down all the resources and prepared a table like the one shown in figure 3, the next step is to keep the relevant resources and reject the un-necessary ones which may not be directly related to what we want to work on our our research objectives. Follow the following steps to do that:

 1. For all the resources listed down, finish 10% of reading of each resource or research paper(first pass reading, we will discuss about it later). If we find it not related to our research objective, we can reject it.

 2. If that resurce is related to our objective and is relevant and important to us, do a complete full pass reading over the paper. From the references, if we find any other relevant reference then mark those in the original paper and add them to the list and repeat the same over this new paper or resource now.

So after this, this is what the final table might look like
<p align="center" width="100%">
	<a  href="https://towardsdatascience.com/how-you-should-read-research-papers-according-to-andrew-ng-stanford-deep-learning-lectures-98ecbd3ccfb3"  title="Redirect to homepage">
		<img width="100%" src="https://raw.githubusercontent.com/saiamrit/technical-blog/master/images/read_papers5.png"> 
<figcaption align = "center">Fig.4 - Table after completion of Literature Survey chart for Pose Estimation</figcaption>
	</a>
</p>
Notice that the 2nd, 4th and 6th resources were important and relevant so we read it in detail but the other oned were not very important or the entire thing was not relevant so we read through some portion of each, whatever was necessary and left the rest.

Such a table can be really useful when we return back to it after some months or years to look for or recall what we have read or the papers we have already looked at and rejected. It helps us to save a lot of time iterating over unnecessary resources and helps us effectively dedicate time to the useful resources.
