// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/al-folio/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/al-folio/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A collection of projects I have worked on or am currently working on.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/al-folio/projects/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "This is a description of the page. You can modify it in &#39;_pages/cv.md&#39;. You can also change or remove the top pdf download button.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/al-folio/cv/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/al-folio/books/";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/al-folio/blog/";
              },
            },{id: "post-inside-bytedance-39-s-video-models-building-a-generative-mllm-planner-from-scratch",
        
          title: "Inside ByteDance&#39;s Video Models: Building a Generative MLLM Planner from Scratch",
        
        description: "ByteDance&#39;s video models (Seedance / Seaweed) are rumored to scale up to 200B parameters. Here is how their generative MLLM planning architecture actually works, and how we built a toy reproduction from scratch.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/al-folio/blog/2026/bernini-mnist-latent-semantic-planning/";
          
        },
      },{id: "post-training-a-waifu-diffusion-model-on-small-corrupted-data-using-patch-diffusion-and-rectified-flow",
        
          title: "Training a Waifu Diffusion Model on Small, Corrupted Data using Patch Diffusion and...",
        
        description: "How to train a data-efficient diffusion model on corrupted anime face data using CIELAB space, patch cropping, and modern transformer techniques.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/al-folio/blog/2026/waifu-diffusion/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/al-folio/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/al-folio/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-inspacemap-platform",
          title: 'InSpaceMap Platform',
          description: "Full-stack indoor venue mapping, drag-and-drop navigation graph editor, and wayfinding platform",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/10_inspacemap/";
            },},{id: "projects-tebak-kata",
          title: 'Tebak Kata',
          description: "Contexto-inspired semantic word-guessing game built with Next.js and vector embedding similarity matching",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/11_tebak_kata/";
            },},{id: "projects-idlerpg-engine",
          title: 'IdleRPG Engine',
          description: "Automated RPG game engine, progression state machine, and battle mechanics backend",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/12_idlerpg/";
            },},{id: "projects-yoshida",
          title: 'Yoshida',
          description: "Production-grade General LLM Reinforcement Learning &amp; Post-Training Framework",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/1_yoshida/";
            },},{id: "projects-axon",
          title: 'Axon',
          description: "A minimal deep learning framework built in C++20 from scratch with eager autograd, AVX2 SIMD CPU backend, and GGML-style quantization",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/2_axon/";
            },},{id: "projects-angel",
          title: 'Angel',
          description: "Next-generation distributed LLM RL framework with VERL-style 3D HybridEngine architecture (In Development)",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/3_angel/";
            },},{id: "projects-flash-sdxl",
          title: 'Flash-SDXL',
          description: "High-throughput end-to-end web serving engine and inference optimization pipeline for SDXL models",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/4_flash_sdxl/";
            },},{id: "projects-project-yoru-amp-fami",
          title: 'Project Yoru &amp;amp; Fami',
          description: "Stealth research on next-generation image generation and speech/audio foundation systems",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/5_yoru_fami/";
            },},{id: "projects-batik-meanflow",
          title: 'Batik MeanFlow',
          description: "Generative Flow Matching and Mean Flow applied to manifold learning and Indonesian Batik pattern synthesis",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/6_batik_meanflow/";
            },},{id: "projects-waifu-diffusion-exploration",
          title: 'Waifu Diffusion Exploration',
          description: "Fine-tuning and latent diffusion conditioning experiments on stylized generative image models",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/7_waifudiffusion/";
            },},{id: "projects-bernini-mnist",
          title: 'Bernini MNIST',
          description: "Generative flow and velocity field modeling experiments on benchmark vision datasets",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/8_bernini_mnist/";
            },},{id: "projects-booru-ecosystem-amp-dataset-harvesting-suite",
          title: 'Booru Ecosystem &amp;amp; Dataset Harvesting Suite',
          description: "Unified dataset ingestion, smart tracking, automated tagging, and scheduling pipeline for mediaboard resources",
          section: "Projects",handler: () => {
              window.location.href = "/al-folio/projects/9_booru_ecosystem/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/al-folio/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/al-folio/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%66%69%78%67%6D%63@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/ruwwww", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/abdurrahman-izzuddin-al-faruq", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
