%
% In dieser Datei fehlen noch ein paar erklaerende Kommentare
%
\ProvidesClass{tuberlinbeamer}
\newcounter{deko}
\setcounter{deko}{0}
\DeclareOption{Nike}{\def\FileName{Nike}\setcounter{deko}{1}}
\DeclareOption{Haus}{\def\FileName{Haus}\setcounter{deko}{2}}
\DeclareOption{Aussicht}{\def\FileName{Aussicht}\setcounter{deko}{3}}
\DeclareOption{Leer}{\def\Filename{}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass[xcolor=table,hyperref={bookmarks},9pt,t]{beamer}  % pdfpagemode=FullScreen
\RequirePackage[scaled=0.85]{helvet}
\RequirePackage{xspace}
\RequirePackage{etoolbox}

\setlength{\unitlength}{1mm}
\newlength{\rand}
\setlength{\rand}{5mm}
% 128 mm x 96 mm

\xdefinecolor{TUred}{RGB}{197,14,31}
\xdefinecolor{grau}{RGB}{113,113,113}
\setbeamersize{text margin left=\rand, text margin right=\rand}

\newcommand*{\lengthinmm}[1]{\strip@pt\dimexpr0.351459804\dimexpr#1\relax\relax}
\newcommand{\smallcirc}{\begin{picture}(2,2)\put(1,0.9){\circle*{0.88}}\end{picture}}
\newcommand{\Nike}{\def\FileName{Nike}\setcounter{deko}{1}}
\newcommand{\Haus}{\def\FileName{Haus}\setcounter{deko}{2}}
\newcommand{\Aussicht}{\def\FileName{Aussicht}\setcounter{deko}{3}}
\newcommand{\Leer}{\setcounter{deko}{0}}
\newcommand{\sehrLeer}{\setcounter{deko}{-1}}
\newcommand{\pagename}{Seite} % Definition wird vom babel-Paket ueberschrieben

\linespread{1.1}
\newcommand{\leer}{}
\newcommand{\pathtologo}{TU_Logo_lang_4c_rot}
\newcommand{\pathtomylogo}{}

\setbeamertemplate{title page}{%
	\thispagestyle{empty}
	\vspace{-\headheight}
	\hspace{-\rand}%
	\begin{picture}(100,93.6)
		\put(93,74){\includegraphics[width=30mm]{\pathtologo}}
		\put(-0.5,32){\ifnum\thedeko>0\color{TUred}\rule{\rand}{32mm}\fi}
		\put(5,32){\ifnum\thedeko>0\includegraphics[height=32mm,width=118mm]{\FileName}\fi}
		\put(5,8){\parbox[b][23mm][s]{118mm}{%
      \raggedright\vfill
      \usebeamerfont{title}\usebeamercolor[fg]{titlelike}{\inserttitle\par}
      \vfill
      \color{grau}\fontsize{8pt}{8pt}\selectfont\insertauthor%
			
			\ifdefempty{\insertinstitute}{}{\insertinstitute}%
			
			\ifdefempty{\insertsubtitle}{}{\insertsubtitle}\par%
      \vfill%
    }}
		\put(5,7.5){\ifx\pathtomylogo\leer%
      {\color{grau}{\rule{118mm}{0.5pt}}}
      \else{\color{grau}{\rule{118mm}{0.5pt}}}
      \fi}
	\end{picture}
}

\setbeamertemplate{headline}{%
	\begin{picture}(100,16)
		\put(-0.5,1){\ifnum\thedeko>0\color{TUred}\rule{\rand}{10mm}\fi}
		\put(5,1){\ifnum\thedeko>0\includegraphics[height=10mm,width=94mm]{\FileName_head}\fi}
		\put(104,1){\ifnum\thedeko>-1\includegraphics[width=19mm]{\pathtologo}\fi}
	\end{picture}
}

\setbeamertemplate{frametitle}{%
	\ifnum\thedeko<0
		\vspace{-12.5mm}
		\insertframetitle
	\fi
	\ifnum\thedeko=0
		\vspace{-6mm}
		\begin{picture}(0,0)
			\put(0,0){\parbox[b][10mm][b]{94mm}{\raggedright\insertframetitle\vphantom{Ag}}}
		\end{picture}
	\fi
	\ifnum\thedeko>0
		\insertframetitle
	\fi
\vspace{-1mm}
}

\setbeamertemplate{footline}{%
	\begin{picture}(100,10)
    \put(5,7.5){\rule{118mm}{0.5pt}}
    \put(5,1){\parbox[b][6mm][s]{95mm}{\fontsize{5pt}{5pt}\selectfont
      \raggedright\vfill
      \textbf{\insertshorttitle}~\textbar~\insertshortauthor
      \ifdefempty{\insertsubtitle}{}{~\textbar~\insertsubtitle}
      \vfill\vfill
      \pagename~\insertframenumber\par\vfill%
    }}%
    \put(123,1){\ifx\pathtomylogo\leer
      \else{\makebox[0mm][r]{\includegraphics[height=6mm]{\pathtomylogo}}}
      \fi}
  \end{picture}
}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{background}{}
\setbeamertemplate{itemize item}{--}
\setbeamertemplate{itemize subitem}{\smallcirc}
\setbeamertemplate{itemize subsubitem}{\frqq}

\setbeamercolor{alerted text}{fg=TUred}
\setbeamercolor{block body}{bg=grau}
\setbeamercolor{block body alerted}{bg=grau}
\setbeamercolor{block title}{bg=TUred,fg=white}
\setbeamercolor{block title alerted}{bg=TUred,fg=white}
\setbeamercolor{block title example}{bg=TUred,fg=white}
\setbeamercolor{frametitle}{fg=TUred}
\setbeamercolor{item}{fg=black}
\setbeamercolor{navigation symbols}{fg=grau}
\setbeamercolor{section in toc}{fg=TUred}
\setbeamercolor{titlelike}{fg=TUred}
\setbeamercolor{footline}{fg=grau}

\setbeamerfont{block title}{size=\normalsize}
\setbeamerfont{block title alerted}{size=\normalsize}
\setbeamerfont{block title example}{size=\normalsize}
\setbeamerfont{titlelike}{}
\setbeamerfont{title}{size=\fontsize{12}{14}\selectfont}
\setbeamerfont{title in head}{series=\bfseries}
\setbeamerfont{frametitle}{size=\large}
\setbeamerfont{section in toc}{series=\bfseries}

\setbeamerfont*{itemize/enumerate body}{parent=normal text}
\setbeamerfont*{itemize/enumerate subbody}{parent=itemize/enumerate body}
\setbeamerfont*{itemize/enumerate subsubbody}{parent=itemize/enumerate body}

\renewcommand{\emph}{\textbf}
\renewcommand{\ttdefault}{lmtt}
\newcommand{\fazit}[1]{{\color{TUred}#1}\xspace}
