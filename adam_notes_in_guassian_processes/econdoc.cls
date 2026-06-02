%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% econdoc.cls -- Companion to econcv.cls for prose documents (SOP, addendum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{econdoc}[2026/05/28 v1.0 Economics PhD prose document class]
\LoadClass[11pt,letterpaper]{article}
\RequirePackage[left=1in,right=1in,top=1in,bottom=0.95in]{geometry}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{hyperref}
\RequirePackage{xcolor}

\definecolor{citepink}{HTML}{D6336C}
\definecolor{linkgreen}{HTML}{2E7D32}
\definecolor{footgray}{gray}{0.45}

\hypersetup{
  colorlinks = true,
  citecolor  = citepink,
  linkcolor  = linkgreen,
  urlcolor   = linkgreen
}

\RequirePackage[round,authoryear]{natbib}
\bibliographystyle{aer}

% ---- Paragraph settings for prose ----
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

% ---- Storage ----
\newcommand{\name}[1]{\def\@name{#1}}
\newcommand{\school}[1]{\def\@school{#1}}
\newcommand{\doctitle}[1]{\def\@doctitle{#1}}
\newcommand{\footer}[1]{\def\@footer{#1}}

% ---- Header: title, school, name ----
\newcommand{\makeheader}{%
  \begin{center}
    {\LARGE\scshape\@doctitle}\\[8pt]
    {\large\@school}\\[4pt]
    {\normalsize\@name}
  \end{center}
  \vspace{6pt}
}

% ---- Footer: grayed contact ----
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyfoot[L]{\small\color{footgray}\@footer}